"""
dataset_generator.py  —  Bloom's Taxonomy Question Generator
=============================================================
Architecture:
  1. Extract rich concepts from the user's input text
  2. Use STRUCTURAL PATTERNS (verb + skeleton) — no domain content
     from the dataset ever bleeds into the output
  3. Combine user concepts with structural patterns to build questions
     that are 100% on-topic

This ensures every generated question is about the user's topic.
"""

import csv
import random
from collections import Counter, defaultdict
import re
import os

# ── Bloom level → dataset category ──────────────────────────────────────────
LEVEL_MAP = {
    "Remember":   "BT1",
    "Understand": "BT2",
    "Apply":      "BT3",
    "Analyze":    "BT4",
    "Evaluate":   "BT5",
    "Create":     "BT6",
}

# ── Dataset path ─────────────────────────────────────────────────────────────
_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(_DIR, "bloom_dataset.csv")


# =============================================================================
# SECTION 1 — Concept extraction
# =============================================================================

_STOPWORDS = {
    'a','an','the','is','are','was','were','be','been','being',
    'have','has','had','do','does','did','will','would','could','should',
    'may','might','shall','can','need','dare','ought',
    'that','this','these','those','it','its','they','them','their',
    'which','who','whom','what','when','where','how','whether',
    'and','or','but','nor','so','yet','for','of','in','on','at',
    'to','from','with','by','about','into','through','during',
    'including','consists','several','various','different','many','some',
    'also','while','whether','follows','follow','within','across',
    'using','uses','use','make','makes','made',
    'each','such','more','most','very','just','then','than','thus',
    'both','all','any','few','over','under','between','among',
    'written','called','known','based','given','found',
    'final','high','level','main','key','next','last','first',
    'certain','specific','general','particular','similar','same',
    'used','produce','produces','produced',
    # extra prepositions/connectors that cause garbage compounds
    'like','simply','directly','similarly','instead','example',
    'however','therefore','hence','thus','whereby','whereas',
    'despite','although','because','since','unless','until',
    'rather','either','neither','whether','another','other',
}

# Technical head nouns to anchor compound extraction
_TECH_HEADS = (
    'analysis','generation','optimization','compilation','execution',
    'processing','recognition','detection','classification','translation',
    'learning','network','algorithm','architecture','design','system',
    'language','code','program','model','structure','phase','stage',
    'token','tokens','parser','scanner','checker','verifier','processor',
    'instruction','instructions','memory','register','machine',
    'grammar','rule','rules','error','errors','output','input',
    'source','target','type','checking','binding','scope','symbol','table',
    'tree','graph','flow','data','stack','queue','heap','buffer','cache',
    'compiler','interpreter','assembler','linker','loader','debugger',
    'module','function','variable','constant','operator','expression',
    'statement','declaration','definition','abstraction','interface',
    'class','object','method','attribute','parameter','argument',
    'loop','condition','branch','recursion','iteration',
    # OS / systems domain
    'management','scheduling','allocation','protection','synchronization',
    'storage','driver','kernel','process','thread','service','resource',
    'access','space','policy','mode','call','interrupt','handler',
    # Networking / DB domain
    'protocol','packet','routing','query','transaction','index','schema',
    'relation','table','key','constraint','integrity','recovery',
    # ML / AI domain
    'training','model','layer','weight','gradient','feature','label',
    'inference','prediction','classification','regression','clustering',
    # Complexity Theory & Algorithms
    'problem','problems','algorithm','algorithms','complexity','verification',
    'certificate','reduction','computation','computability','decidability',
    'tractability','feasibility','approximation','heuristic',
    # Software Engineering
    'pattern','patterns','testing','requirement','requirements',
    'architecture','design','lifecycle','methodology','refactoring',
    # Computer Architecture
    'pipeline','register','cache','processor','instruction','bus',
    'interrupt','microprocessor','microcontroller','firmware',
    # Cybersecurity
    'encryption','decryption','authentication','vulnerability','exploit',
    'cipher','malware','firewall','cryptography','protocol',
    # Theory of Computation
    'automaton','automata','grammar','language','reduction',
    'decidability','computability', 'transducer',
    # OOP
    'inheritance','polymorphism','encapsulation','abstraction',
    'class','object','interface','constructor','method',
    # Cloud & Web
    'container','microservice','virtualization','serverless',
    'api','endpoint','middleware','deployment','scalability',
    # Parallel & Distributed
    'thread','mutex','synchronization','parallelism','concurrency',
    'deadlock','semaphore','barrier','latency','throughput',
    # Data Structures & Algorithms domain
    'element','elements','operation','operations','format','relationship',
    'traversal','insertion','deletion','searching','sorting','hashing',
    'pointer','node','nodes','edge','edges','path','paths','list','array',
    'arrays','tree','trees','graph','graphs','heap','stack','queue',
    'matrix','set','map','pair','entry','entries','record','records',
    'complexity','notation','performance','efficiency',
    'retrieval','organization','arrangement','storage','access',
    'usage','utilization','implementation','representation',
)


def extract_concepts(text):
    """
    Extract domain-specific noun phrases from user input.
    Returns list ordered best-first (multi-word technical > single technical).
    """
    raw = []

    # Normalise text:
    # 1. Strip possessive apostrophes (straight and curly): "computer's" → "computer"
    clean_text = re.sub(r'[\u2019\u2018\']s\b', '', text)   # smart + straight apostrophe + s
    clean_text = re.sub(r'[\u2019\u2018\']', '', clean_text)  # any remaining
    # 2. Replace commas, semicolons and conjunctions with spaces
    clean_text = re.sub(r'[,;]', ' ', clean_text)
    clean_text = re.sub(r'\band\b|\bor\b|\bbut\b', ' ', clean_text, flags=re.I)

    # Pass 1: strict technical compounds  e.g. "lexical analysis", "code generation"
    # Pattern: exactly ONE clean modifier word + tech head
    head_re = (r'(?<![A-Za-z])([A-Za-z]+)'          # exactly one modifier word
               r'\s+(' + '|'.join(_TECH_HEADS) + r')\b')
    for m in re.finditer(head_re, clean_text, re.I):
        modifier = m.group(1).strip().lower()
        head     = m.group(2).strip().lower()
        # Reject if modifier is a stopword
        if modifier in _STOPWORDS:
            continue
        # Reject comparative/superlative adjectives as modifiers
        _COMPARATIVE_ADJ = {
            'faster','slower','larger','smaller','better','worse',
            'higher','lower','simpler','harder','easier','bigger',
            'greater','lesser','wider','narrower','deeper','shorter',
            'longer','stronger','weaker','older','newer','closer',
            'fast','slow','quick','direct','random','sequential',
            'contiguous','hierarchical','circular','doubly','singly',
        }
        if modifier in _COMPARATIVE_ADJ:
            continue
        # Reject if modifier looks like a verb (covers organizes, handles, manages, etc.)
        _VERB_ENDINGS = ('ing','tes','ted','cts','ces','ves','ied',
                         'zes','ses','nes','ges','les','res','kes',
                         'nds','rds','lps','mps','ngs','rks','lts')
        if any(modifier.endswith(s) for s in _VERB_ENDINGS):
            continue
        # Reject verb-s forms: manages, reduces, performs etc.
        _VERB_ROOTS = {
            'organize','manage','handle','provide','require','contain',
            'include','define','allow','enable','control','determine',
            'produce','create','perform','execute','schedule','terminate',
            'allocate','deallocate','swap','convert','compile','translate',
            'parse','scan','check','verify','detect','generate','store',
            'retrieve','access','process','compute','communicate','ensure',
            'delete','update','insert','select','query','reduce','discover',
            'measure','predict','classify','cluster','transform','encode',
            'decode','compress','encrypt','decrypt','route','forward',
            'transmit','receive','send','connect','establish','maintain',
            'monitor','evaluate','optimize','improve','extend','support',
            'implement','represent','describe','demonstrate','compare',
            'analyze','propose','develop','construct','identify','extract',
            'assign','remove','replace','combine','split','merge','sort',
            'initialize','synchronize','coordinate','validate','normalize',
            'index','partition','migrate','backup','restore','recover',
            'replicate','distribute','apply','use','make','take','give',
            'store','access','follow','allow','consist','contain','represent',
        }
        if (modifier.endswith('s') and len(modifier) > 4
                and not modifier.endswith(('ness','ics','ss','us','is'))):
            root = modifier[:-1]
            if root in _VERB_ROOTS:
                continue
            if modifier.endswith('es') and modifier[:-2] in _VERB_ROOTS:
                continue
        raw.append(modifier + ' ' + head)

    # Filter known false compounds (verb-as-modifier or nonsense pairings)
    _FALSE_COMPOUNDS = {
        'optimization code', 'generated code', 'machine code',
        'grammatical rules', 'software program', 'high language',
        'level language', 'final instructions',
        # OS domain false hits
        'organizes data', 'manages memory', 'handles process',
        'provides service', 'allows process', 'involves process',
        'determines process', 'swapping process', 'controls access',
        'physical memory', 'common service', 'main function',
        'management memory',
        'include process', 'gets access', 'write access', 'read access',
        'given time', 'disk space', 'main memory', 'hardware device',
        'software resource', 'computer program', 'common service',
        # Networking false hits
        'manage network', 'ensure data', 'control network', 'forward packet',
        'enable communication', 'establish connection', 'maintain state',
        'handle error', 'detect error', 'correct error', 'send packet',
        # Database false hits
        'delete data', 'update data', 'insert data', 'select data',
        'organize data', 'ensure integrity', 'reduce redundancy',
        'improve performance', 'manage transaction', 'create index',
        # ML false hits
        'perform classification', 'perform regression', 'measure error',
        'measure prediction', 'update weight', 'minimize loss',
        'discover pattern', 'reduce dimension', 'address overfitting',
        'error gradient', 'predict output', 'improve accuracy',
        # Data Structures false hits
        'speed memory', 'faster searching', 'large dataset', 'logical manner',
        'crucial component', 'efficient software', 'direct impact',
        'simply dumping', 'large datasets', 'faster access', 'faster retrieval',
        'like arrays', 'lists trees', 'like trees', 'like graphs', 'like lists',
        'similarly computer', 'manage large', 'manage data',
        # Example/context phrases — not core technical concepts
        'social media', 'road maps', 'road map', 'media connections',
        # DS false attribute / process phrases
        'hierarchical structure', 'hierarchical structures', 'efficient searching',
        'call management', 'explore nodes', 'simplest data', 'search trees',
        'linear data', 'non-linear', 'sorted order', 'expression evaluation',
        'function call', 'depth-first', 'breadth-first',
        # Blockchain false hits
        'certain blockchain', 'various industries', 'chronological order',
        # DS pattern false hits (verb+noun pairs)
        'deletion searching', 'include arrays', 'include trees', 'include graphs',
        'include lists', 'insertion deletion', 'searching operations',
        'store elements', 'as nodes', 'fast access', 'slow access',
        'allow access', 'modified efficiently', 'accessed modified',
        'consist vertices', 'represent networks', 'added removed',
        'high energy', 'actively working', 'widely known', 'single centralized',
        'chain management', 'regulatory concerns', 'distributed copies',
    }
    raw = [c for c in raw if c not in _FALSE_COMPOUNDS]

    # Direct injection: known important multi-word phrases found verbatim in text
    _KNOWN_PHRASES = [
        # Compiler / PL
        'code generation', 'type checking', 'symbol table', 'parse tree',
        'syntax tree', 'abstract syntax', 'intermediate code', 'target code',
        'source code', 'object code', 'error detection', 'error recovery',
        'data flow', 'control flow', 'dead code', 'lexical analysis',
        'syntax analysis', 'semantic analysis', 'code optimization',
        # OS / Systems
        'process management', 'memory management', 'file system management',
        'device management', 'virtual memory', 'file system', 'page table',
        'page fault', 'context switch', 'system call', 'device driver',
        'round robin', 'priority scheduling', 'cpu scheduling', 'cpu scheduler',
        'memory allocation', 'memory deallocation', 'address space',
        'disk storage', 'process scheduling', 'process synchronization',
        'mutual exclusion', 'deadlock detection', 'deadlock prevention',
        'semaphore', 'critical section', 'operating system',
        # Networking
        'ip address', 'mac address', 'tcp ip', 'network protocol',
        'packet switching', 'routing table', 'domain name', 'subnet mask',
        # Database
        'primary key', 'foreign key', 'sql query', 'relational database',
        'normal form', 'entity relationship', 'transaction management',
        'data integrity', 'index structure', 'query optimization',
        # ML / AI
        'neural network', 'deep learning', 'machine learning', 'training data',
        'loss function', 'gradient descent', 'feature extraction',
        'decision tree', 'random forest', 'support vector',
        'natural language', 'convolutional network', 'recurrent network',
        'supervised learning', 'unsupervised learning', 'reinforcement learning',
        'activation function', 'hidden layer', 'output layer', 'input layer',
        'back propagation', 'forward propagation', 'dropout regularization',
        'model evaluation', 'cross validation', 'confusion matrix',
        # Networking
        'application layer', 'transport layer', 'network layer', 'data link layer',
        'physical layer', 'session layer', 'presentation layer',
        'tcp ip', 'ip address', 'mac address', 'subnet mask', 'routing table',
        'congestion control', 'flow control', 'error detection', 'error correction',
        'packet switching', 'circuit switching', 'sliding window',
        'domain name', 'dns server', 'dhcp server', 'network protocol',
        'osi model', 'tcp model',
        # Database
        'database management', 'relational database', 'sql query',
        'primary key', 'foreign key', 'normal form', 'first normal form',
        'second normal form', 'third normal form', 'entity relationship',
        'transaction management', 'data integrity', 'query optimization',
        'index structure', 'b tree', 'hash index', 'concurrency control',
        'locking protocol', 'timestamp ordering', 'acid properties',
        'structured query', 'query language',
        # Data Structures & Algorithms
        'data structure', 'data structures', 'data element', 'data elements',
        'array', 'arrays', 'stack', 'stacks', 'queue', 'queues',
        'tree', 'trees', 'graph', 'graphs', 'vertex', 'vertices',
        'node', 'nodes', 'root node', 'child node', 'child nodes',
        'last in first out', 'first in first out', 'lifo', 'fifo',
        'contiguous memory', 'pointer arithmetic', 'index access',
        'social media', 'road map', 'road maps',
        'data type', 'data types', 'data model', 'data access', 'data storage',
        'data retrieval', 'data organization', 'data management',
        'linked list', 'linked lists', 'doubly linked', 'circular list',
        'dynamic array', 'dynamic arrays', 'static array', 'static arrays',
        'binary tree', 'binary trees', 'binary search', 'search tree',
        'balanced tree', 'binary heap', 'avl tree',
        'heap sort', 'merge sort', 'quick sort', 'bubble sort', 'insertion sort',
        'depth first', 'breadth first', 'graph traversal',
        'hash table', 'hash tables', 'hash function', 'hash map', 'hash collision',
        'stack overflow', 'queue operations', 'priority queue',
        'time complexity', 'space complexity', 'big notation',
        'memory usage', 'memory management', 'memory allocation', 'memory address',
        'pointer arithmetic', 'dynamic memory', 'static memory',
        'tree traversal', 'tree rotation', 'tree balancing',
        'graph algorithm', 'shortest path', 'minimum spanning',
        'adjacency matrix', 'adjacency list', 'edge weight',
        'algorithm performance', 'algorithm efficiency', 'algorithm complexity',
        'software application', 'software system', 'software component',
        'array access', 'array storage', 'sequential access', 'random access',
        'hierarchical data', 'hierarchical structure',
        # Complexity Theory / Algorithms
        'computational complexity', 'complexity theory', 'complexity class',
        'complexity class p', 'complexity class np',
        'polynomial time', 'polynomial time algorithm', 'polynomial time algorithms',
        'exponential time', 'factorial time',
        'np-complete', 'np-hard', 'np complete', 'np hard',
        'non-deterministic polynomial time', 'non-deterministic polynomial',
        'tractable problem', 'tractable problems',
        'intractable problem', 'intractable problems',
        'easy problem', 'easy problems',
        'hard problem', 'hard problems',
        'deterministic algorithm', 'deterministic algorithms',
        'non-deterministic algorithm', 'non-deterministic algorithms',
        'clique problem', 'travelling salesman problem', 'traveling salesman problem',
        'hamiltonian cycle problem', 'hamiltonian cycle',
        'graph coloring problem', 'graph coloring', 'subset sum problem', 'subset sum',
        'verification process', 'decision problem', 'decision version',
        'input size', 'running time', 'time complexity', 'space complexity',
        'merge sort', 'quick sort', 'binary search',
        'breadth-first search', 'depth-first search',
        'dijkstra algorithm', 'kruskal algorithm', 'prim algorithm',
        'minimum spanning tree', 'shortest path algorithm',
        # Blockchain / Distributed Systems
        'blockchain technology', 'distributed ledger', 'digital ledger',
        'supply chain', 'supply chain management', 'smart contract', 'smart contracts',
        'digital identity', 'digital currency', 'digital asset',
        'data integrity', 'data transparency', 'data immutability',
        'consensus mechanism', 'proof of work', 'proof of stake',
        'energy consumption', 'scalability issues', 'regulatory concerns',
        'voting systems', 'voting system', 'healthcare systems',
        'network participants', 'network nodes', 'peer network',
        'transaction records', 'transaction data', 'transaction validation',
        'block structure', 'block chain', 'hash function', 'hash value',
        'public key', 'private key', 'digital signature',
        'decentralized network', 'centralized database', 'distributed network',
        'immutability', 'transparency', 'decentralization',
        # OS — extended
        'paging', 'segmentation', 'thrashing', 'spooling', 'buffering',
        'memory fragmentation', 'internal fragmentation', 'external fragmentation',
        'demand paging', 'page replacement', 'page replacement algorithm',
        'least recently used', 'first in first out', 'optimal page replacement',
        'process state', 'process control block', 'inter-process communication',
        'pipe', 'message queue', 'shared memory', 'monitor',
        'readers writers problem', 'producer consumer problem', 'dining philosophers',
        'starvation', 'priority inversion', 'banker algorithm',
        'disk scheduling', 'disk management', 'file allocation',
        'contiguous allocation', 'linked allocation', 'indexed allocation',
        'inode', 'directory structure', 'file permission',
        # Networks — extended
        'three-way handshake', 'tcp handshake', 'four-way handshake',
        'network address translation', 'nat', 'vpn', 'firewall',
        'http', 'https', 'ftp', 'smtp', 'pop', 'imap',
        'socket', 'port number', 'ip datagram', 'frame',
        'arp protocol', 'icmp protocol', 'rip protocol', 'ospf protocol',
        'distance vector', 'link state', 'routing algorithm',
        'bandwidth', 'latency', 'throughput', 'jitter',
        'network topology', 'bus topology', 'star topology', 'ring topology',
        'ipv4 address', 'ipv6 address', 'subnetting', 'subnet',
        'classful addressing', 'classless addressing', 'cidr',
        'multiplexing', 'demultiplexing', 'encapsulation',
        'network security', 'ssl', 'tls', 'certificate authority',
        # Databases — extended
        'er diagram', 'entity relationship diagram', 'entity', 'attribute',
        'database view', 'stored procedure', 'trigger', 'cursor',
        'nosql', 'mongodb', 'document database', 'key value store',
        'column family', 'graph database', 'document store',
        'cap theorem', 'eventual consistency', 'strong consistency',
        'sharding', 'replication', 'partitioning',
        'join operation', 'inner join', 'outer join', 'natural join',
        'aggregate function', 'group by', 'having clause',
        'functional dependency', 'partial dependency', 'transitive dependency',
        'boyce codd normal form', 'bcnf',
        'data warehouse', 'data mining', 'olap', 'oltp',
        'relational model', 'object relational', 'xml database',
        # Compiler Design — extended
        'finite automaton', 'deterministic finite automaton', 'nondeterministic finite automaton',
        'dfa', 'nfa', 'regular expression', 'regular language',
        'context free grammar', 'cfg', 'pushdown automaton', 'pda',
        'll parser', 'lr parser', 'slr parser', 'lalr parser',
        'shift reduce', 'reduce reduce', 'first set', 'follow set',
        'ambiguous grammar', 'leftmost derivation', 'rightmost derivation',
        'syntax directed translation', 'attribute grammar',
        'three address code', 'quadruple', 'triple',
        'register allocation', 'instruction selection', 'peephole optimization',
        'loop optimization', 'constant folding', 'constant propagation',
        'common subexpression', 'strength reduction',
        # ML / AI — extended
        'convolutional neural network', 'cnn', 'recurrent neural network', 'rnn',
        'long short term memory', 'lstm', 'transformer model', 'attention mechanism',
        'self-attention', 'multi-head attention', 'encoder decoder',
        'overfitting', 'underfitting', 'regularization', 'dropout',
        'batch normalization', 'learning rate', 'epoch', 'batch size',
        'precision', 'recall', 'f1 score', 'roc curve', 'auc',
        'k-nearest neighbor', 'naive bayes', 'logistic regression',
        'principal component analysis', 'pca', 'dimensionality reduction',
        'generative adversarial network', 'gan', 'variational autoencoder', 'vae',
        'transfer learning', 'fine tuning', 'pretrained model',
        'natural language processing', 'nlp', 'word embedding', 'word2vec',
        'tokenization', 'stemming', 'lemmatization', 'named entity recognition',
        'object detection', 'image segmentation', 'semantic segmentation',
        # Software Engineering
        'software development', 'software engineering', 'software lifecycle',
        'software development lifecycle', 'sdlc', 'waterfall model', 'agile model',
        'scrum', 'sprint', 'kanban', 'extreme programming', 'xp',
        'requirements engineering', 'software requirements', 'use case',
        'software design', 'software architecture', 'design pattern',
        'creational pattern', 'structural pattern', 'behavioral pattern',
        'singleton pattern', 'factory pattern', 'observer pattern',
        'mvc pattern', 'model view controller',
        'software testing', 'unit testing', 'integration testing', 'system testing',
        'black box testing', 'white box testing', 'regression testing',
        'test case', 'test suite', 'code coverage',
        'software maintenance', 'software quality', 'software metric',
        'uml diagram', 'class diagram', 'sequence diagram', 'activity diagram',
        'object oriented', 'object oriented design', 'solid principle',
        'version control', 'git', 'continuous integration', 'continuous deployment',
        # Computer Architecture
        'central processing unit', 'cpu', 'arithmetic logic unit', 'alu',
        'control unit', 'register', 'program counter', 'stack pointer',
        'instruction set', 'instruction set architecture', 'isa',
        'risc', 'cisc', 'pipeline', 'pipelining', 'pipeline stage',
        'instruction fetch', 'instruction decode', 'execute stage',
        'cache memory', 'cache hit', 'cache miss', 'cache replacement',
        'primary cache', 'secondary cache', 'cache coherence',
        'memory hierarchy', 'memory bus', 'address bus', 'data bus',
        'ram', 'rom', 'dram', 'sram', 'flash memory',
        'virtual address', 'physical address', 'address translation',
        'branch prediction', 'out of order execution', 'superscalar',
        'interrupt handler', 'interrupt service routine', 'dma',
        'input output', 'io controller', 'memory mapped io',
        'multicore processor', 'symmetric multiprocessing',
        # Cybersecurity
        'cryptography', 'encryption', 'decryption', 'cipher',
        'symmetric encryption', 'asymmetric encryption', 'public key cryptography',
        'rsa algorithm', 'aes algorithm', 'des algorithm',
        'digital signature', 'message authentication', 'hash function',
        'md5', 'sha', 'sha-256', 'checksum',
        'authentication', 'authorization', 'access control',
        'multi-factor authentication', 'single sign on', 'identity management',
        'sql injection', 'cross site scripting', 'xss', 'buffer overflow',
        'denial of service', 'distributed denial of service', 'ddos',
        'man in the middle', 'phishing attack', 'ransomware',
        'intrusion detection', 'intrusion prevention', 'ids', 'ips',
        'penetration testing', 'vulnerability assessment', 'risk assessment',
        'security policy', 'security protocol', 'security audit',
        'malware', 'virus', 'trojan', 'worm', 'spyware', 'keylogger',
        # Theory of Computation
        'finite automaton', 'finite state machine', 'turing machine',
        'regular language', 'context free language', 'context sensitive language',
        'chomsky hierarchy', 'formal language', 'formal grammar',
        'pushdown automaton', 'linear bounded automaton',
        'decidable problem', 'undecidable problem', 'semi-decidable',
        'halting problem', 'reduction', 'many-one reduction',
        'computability', 'computable function', 'church turing thesis',
        'pumping lemma', 'closure property',
        # OOP
        'object oriented programming', 'class', 'object', 'instance',
        'inheritance', 'polymorphism', 'encapsulation', 'abstraction',
        'method overriding', 'method overloading', 'constructor',
        'interface', 'abstract class', 'concrete class',
        'single inheritance', 'multiple inheritance', 'multilevel inheritance',
        'static method', 'instance method', 'access modifier',
        'public', 'private', 'protected', 'getter', 'setter',
        'composition', 'aggregation', 'association', 'dependency',
        'coupling', 'cohesion', 'solid principles',
        # Web Technologies
        'html', 'css', 'javascript', 'dom', 'document object model',
        'http protocol', 'https protocol', 'rest api', 'restful api',
        'web service', 'soap', 'json', 'xml',
        'client server', 'web server', 'web browser', 'web application',
        'frontend', 'backend', 'full stack', 'middleware',
        'cookie', 'session', 'token', 'oauth', 'jwt',
        'database connection', 'orm', 'mvc framework',
        'react', 'angular', 'node js', 'express',
        # Cloud Computing
        'cloud computing', 'cloud service', 'cloud model',
        'infrastructure as a service', 'iaas',
        'platform as a service', 'paas',
        'software as a service', 'saas',
        'public cloud', 'private cloud', 'hybrid cloud',
        'virtualization', 'virtual machine', 'hypervisor',
        'container', 'docker', 'kubernetes', 'container orchestration',
        'microservice', 'microservices architecture', 'service mesh',
        'serverless', 'function as a service', 'faas',
        'load balancing', 'auto scaling', 'elastic scaling',
        'cloud storage', 'object storage', 'block storage',
        # Parallel & Distributed Computing
        'parallel computing', 'distributed computing', 'concurrent programming',
        'thread', 'process thread', 'multithreading', 'multiprocessing',
        'mutex', 'lock', 'deadlock', 'race condition', 'thread safety',
        'synchronization', 'barrier', 'condition variable',
        'shared memory model', 'message passing model', 'mpi',
        'mapreduce', 'hadoop', 'spark', 'parallel algorithm',
        'data parallelism', 'task parallelism', 'pipeline parallelism',
        'amdahl law', 'gustafson law', 'speedup', 'efficiency',
        'distributed system', 'fault tolerance', 'replication',
        # Information Retrieval
        'information retrieval', 'search engine', 'web crawling',
        'inverted index', 'posting list', 'tf-idf', 'term frequency',
        'inverse document frequency', 'cosine similarity',
        'relevance ranking', 'pagerank', 'link analysis',
        'precision', 'recall', 'f-measure', 'mean average precision',
        'query processing', 'query expansion', 'boolean retrieval',
        'vector space model', 'probabilistic model', 'language model',
    ]
    for phrase in _KNOWN_PHRASES:
        # Match phrase or its simple plural (phrase + optional s)
        if re.search(r'\b' + re.escape(phrase) + r's?\b', text, re.I):
            # Always inject the canonical (non-plural) form
            canonical = phrase.rstrip('s') if phrase.endswith('s') else phrase
            raw.append(phrase)   # keep original canonical phrase as written

    for head in _TECH_HEADS:
        if re.search(r'\b' + re.escape(head) + r'\b', text, re.I):
            raw.append(head)

    # Pass 3: other long content nouns — strictly filter out verbs
    _BAD_ENDINGS  = ('ing','ize','ise','ify','erts','ects','oves','uces',
                     'ates','ated','fies','verts','duces','hers','ches',
                     # verb -es / -s forms
                     'zes','nes','ges','les','res','kes','nds','rks',
                     'ules','oles','ires','ures','ives','aves','eves')
    _GOOD_ENDINGS = ('tion','ment','ness','ity','ism','ics','sor',
                     'tor','zer','age','ure','er','or','ism','ist')
    # Explicit verb blacklist for words that slip through suffix rules
    _VERB_BLACKLIST = {
        'organizes','manages','handles','provides','requires','contains',
        'includes','defines','allows','enables','controls','determines',
        'produces','creates','performs','executes','schedules','terminates',
        'allocates','deallocates','swapping','converts','compiles','translates',
        'parses','scans','checks','verifies','detects','generates','stores',
        'retrieves','accesses','processes','computes','communicates','involves',
        'swaps','pages','writes','reads','serves','connects','transfers',
        # networking
        'routes','forwards','transmits','receives','sends','establishes',
        'maintains','monitors','encrypts','decrypts','compresses','encodes',
        # database
        'deletes','updates','inserts','selects','queries','normalizes',
        'indexes','partitions','replicates','distributes','ensures','reduces',
        # ML
        'predicts','classifies','clusters','transforms','evaluates','trains',
        'discovers','measures','applies','minimizes','maximizes','updates',
        'overcomes','addresses','improves','extends','represents',
        # Data structures
        'organizes','retrieves','storing','organizing','retrieving',
        'performs','defines','specifies','relates','traverses',
        'inserts','deletes','searches','sorts','hashes','balances',
        'include','includes','deletion','insertion','searching',
    }
    for m in re.finditer(r'\b([A-Za-z]{3,})\b', text):   # lowered min from 5→3 to catch short DS terms
        w = m.group(1).lower()
        if len(w) < 4:   # but skip 1-3 letter words entirely
            continue
        w = m.group(1).lower()
        if w in _STOPWORDS or w in _VERB_BLACKLIST:
            continue
        bad  = any(w.endswith(e) for e in _BAD_ENDINGS)
        good = any(w.endswith(e) for e in _GOOD_ENDINGS)
        if good or not bad:
            raw.append(w)

    # Deduplicate: keep all multi-word compounds (even if sharing a head),
    # remove single words that are fully contained in a multi-word entry.
    # Treat plural and singular as the same concept — keep the shorter form.
    seen, result = set(), []

    def _canonical(s):
        """Return singular canonical form for dedup purposes."""
        s = s.strip().lower()
        # Remove trailing 's' only if it makes a meaningful plural
        # e.g. "data structures" → "data structure", but "analysis" → "analysis"
        if s.endswith('s') and not s.endswith(('ss','us','is','ics','ness')):
            return s[:-1]
        return s

    # First: add all multi-word compounds
    for c in sorted(set(raw), key=len, reverse=True):
        c_norm = c.strip().lower()
        if not c_norm:
            continue
        words = c_norm.split()
        if len(words) > 1:
            if any(w in _STOPWORDS for w in words):
                continue
            key = _canonical(c_norm)   # treat plural/singular as same
            if key not in seen:
                seen.add(key)
                result.append(c_norm)
    # Then: add single-word concepts — only if they are genuine domain nouns
    # (must appear in _TECH_HEADS, or end in a strong noun suffix)
    _STRONG_NOUN_ENDINGS = ('tion','ment','ness','ity','ism','ics',
                             'sor','tor','zer','ism','ist','ory','ery')
    tech_head_set = set(_TECH_HEADS)
    for c in sorted(set(raw), key=len, reverse=True):
        c_norm = c.strip().lower()
        words  = c_norm.split()
        if len(words) == 1:
            if c_norm in _STOPWORDS:
                continue
            if c_norm not in seen:
                is_tech   = (c_norm in tech_head_set or
                             _canonical(c_norm) in tech_head_set)
                is_noun   = any(c_norm.endswith(e) for e in _STRONG_NOUN_ENDINGS)
                if is_tech or is_noun:
                    seen.add(c_norm)
                    result.append(c_norm)

    # Remove adjectives, adverbs, and overly generic single-word concepts
    _NON_NOUNS = {
        # adjectives
        'grammatical','computational','efficient','digital','logical',
        'physical','virtual','internal','external','written','executed',
        'generated','final','available','possible','specific','multiple',
        'additional','different','various','important','necessary',
        'common','given','following','current','previous','actual',
        'typical','standard','basic','primary','secondary','initial',
        'crucial','efficient','effective','logical','specialized',
        # adverbs
        'physically','logically','digitally','actually','typically',
        'generally','usually','normally','directly','automatically',
        'essentially','effectively','efficiently','initially','currently',
        # overly generic nouns — meaningful only inside compounds
        'information','utilization','performance','searching',
        'manner','choice','component','example','relationship','impact',
        'operation','approach','method','technique','concept','aspect',
        'feature','property','characteristic','behavior','purpose',
        'limitation','advantage','disadvantage','difference','similarity',
        'collection','principle','position','order','rear','front',
    }
    result = [c for c in result if c not in _NON_NOUNS]

    # Re-sort: DS structure names (array, stack, tree, etc.) should come
    # BEFORE generic attribute compounds (contiguous memory, hierarchical data)
    # so they make it through the cap and into Pool A.
    _ds_singles = [c for c in result if c.lower() in _DS_STRUCTURE_NAMES]
    _other      = [c for c in result if c.lower() not in _DS_STRUCTURE_NAMES]
    result = _other[:8] + _ds_singles + _other[8:]  # interleave after top multi-words

    # Final filter: remove known low-value or example-context concepts
    _EXAMPLE_CONTEXT = {
        'social media', 'road maps', 'road map', 'media connections',
        'road network', 'last in', 'first out', 'first in',
        # Complexity theory garbage compounds from general text
        'problem graph', 'include sorting', 'time space', 'every pair',
        'large input', 'time algorithm', 'theoretical model',
        'include binary', 'include merge', 'include breadth',
        'minimum spanning',   # keep as known phrase, remove as raw compound
        'input size',         # keep as known phrase
        'running time',       # keep as known phrase
        # Complexity theory noise phrases
        'theoretical model', 'problem graph', 'include sorting',
        'time space', 'every pair', 'large input', 'time algorithm',
        'include binary', 'conventional computing',
        # DS process/attribute phrases that aren't standalone concepts
        'hierarchical structure', 'hierarchical structures',
        'efficient searching', 'call management', 'explore nodes',
        'simplest data', 'search trees', 'linear data',
        'expression evaluation', 'function call',
    }
    result = [c for c in result if c.lower() not in _EXAMPLE_CONTEXT]

    return result[:25] if result else [
        "the system", "the process", "the concept", "the model", "the algorithm"
    ]


# =============================================================================
# SECTION 1b — Relationship extraction
# =============================================================================

def extract_relationships(text, concepts):
    """
    Find concept pairs that co-occur within the same sentence in the text.
    Returns a list of (A, B) tuples ordered by co-occurrence frequency.
    These pairs are used to bias {A}/{B} slot filling toward text-grounded pairings.
    """
    # Split into sentences
    sentences = re.split(r'[.!?]', text.lower())
    concept_lower = [c.lower() for c in concepts]

    pair_counts = Counter()
    for sent in sentences:
        present = [c for c in concept_lower if c in sent]
        # Record all pairs present in the same sentence
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                a, b = present[i], present[j]
                pair_counts[(a, b)] += 1
                pair_counts[(b, a)] += 1

    # Return pairs sorted by frequency, using original casing
    lower_to_orig = {c.lower(): c for c in concepts}
    pairs = []
    seen = set()
    for (a_l, b_l), count in pair_counts.most_common():
        if (b_l, a_l) not in seen and a_l in lower_to_orig and b_l in lower_to_orig:
            pairs.append((lower_to_orig[a_l], lower_to_orig[b_l], count))
            seen.add((a_l, b_l))
    return pairs


# =============================================================================
# SECTION 2 — Structural patterns
# =============================================================================
# Pure structural skeletons — {A} and {B} are filled with the user's concepts.
# No domain content from the dataset leaks into the output.

_STRUCTURAL_PATTERNS = {
    # ── BT1 Remember ── verbs: define, list, identify, name ──────────────────
    # Each pattern asks for ONE specific answerable fact — no vague scope.
    "BT1": [
        # single-concept
        "Define {A}.",
        "Define {A} and state what type of structure it is.",
        "List two common applications of {A}.",
        "List the key operations supported by {A} and their typical time complexity.",
        "Identify the primary purpose of {A} and what problem it solves.",
        "Identify one situation where {A} would NOT be the appropriate choice.",
        "Name the key property of {A} and give one example of where it applies.",
        "Name the main structural or behavioural difference between {A} and a simpler alternative.",
        # two-concept: only for same-tier, text-grounded pairs
        "Identify which of {A} or {B} is more efficient and in what type of problem.",
        "Name one situation where {A} is preferred over {B} and explain why briefly.",
        "List one key difference between {A} and {B} in terms of memory usage.",
        "Identify the structural difference between {A} and {B}.",
    ],
    # ── BT2 Understand ── verbs: explain, describe, summarize ─────────────────
    # Each pattern asks for interpretation. Two-concept patterns use same-tier
    # pairs only to avoid false technical relationships.
    "BT2": [
        # single-concept
        "Explain how {A} works and what problem it is designed to solve.",
        "Explain what {A} is and describe the key property that makes it useful.",
        "Describe the internal structure of {A} and how it enables its key operations.",
        "Describe the trade-offs of using {A} in terms of time and space complexity.",
        "Summarize when {A} is the appropriate choice and when it is not.",
        "Summarize what {A} is and explain the key trade-offs involved in using it.",
        # two-concept: same-tier, directional
        "Explain why {A} is more efficient than {B} for the same class of problems.",
        "Explain the difference between {A} and {B} in terms of their purpose, behaviour, and limitations.",
        "Describe how {A} and {B} differ in their approach, constraints, and typical use case.",
        "Describe a scenario where {A} would be chosen over {B} and justify the choice.",
        "Summarize the trade-offs between {A} and {B} for a task that requires frequent updates.",
        "Summarize how {A} and {B} each handle the same operation differently.",
    ],
    # ── BT3 Apply ── verbs: demonstrate, illustrate, use ─────────────────────
    # Concrete operational scenarios — no vague phrases like "demanding conditions"
    # or "a specific problem" without defining it.
    "BT3": [
        # single-concept with concrete scenario
        "Demonstrate how {A} is applied to a specific scenario by walking through each step.",
        "Demonstrate how {A} is used in practice by working through a small example.",
        "Demonstrate how {A} processes an operation by walking through a step-by-step example.",
        "Illustrate the step-by-step process of applying {A} to a small concrete example.",
        "Illustrate how {A} processes a specific input by tracing through each step of its operation.",
        "Illustrate how {A} works by showing a concrete scenario where it is applied.",
        "Use {A} to show how a specific type of problem can be approached and solved.",
        "Use {A} to solve a specific instance of a problem, showing each step of your reasoning.",
        # two-concept: contrast in concrete scenario
        "Use {A} and {B} to show how each handles the same dataset differently.",
        "Demonstrate the difference between {A} and {B} by applying both to the same problem.",
        "Illustrate a scenario where {A} performs better than {B} and explain the reason.",
        "Use a step-by-step example to show when {A} is a better choice than {B}.",
    ],
    # ── BT4 Analyze ── verbs: compare, analyze, examine ──────────────────────
    # Comparisons only between same-tier structures. Analysis must demand
    # structural reasoning — time complexity, memory, access patterns.
    "BT4": [
        # single-concept deep analysis
        "Analyze the key operations of {A}, how each works, and what determines their efficiency.",
        "Analyze the advantages and disadvantages of {A} compared to an alternative approach.",
        "Examine the internal design of {A} and explain how each part supports its purpose.",
        "Examine the conditions under which {A} outperforms alternatives and when it does not.",
        # two-concept: same-tier structural comparison
        "Compare {A} and {B} in terms of how they work, their advantages, and their limitations.",
        "Compare {A} and {B} in terms of what they require, what they guarantee, and when each is preferred.",
        "Analyze why a developer would choose {A} over {B} and what the deciding factors are.",
        "Analyze the structural difference between {A} and {B} and its effect on performance.",
        "Examine how {A} and {B} handle the same operation and explain which is more efficient.",
        "Examine the situations where {A} is a better choice than {B} and justify your answer.",
        # challenge-only patterns
        "Analyze the causes and consequences of {A} and explain their impact on {B}.",
        "Examine the main factors that contribute to {A} and how they limit {B}.",
        "Compare the trade-offs of {A} versus {B} and determine which is more suitable.",
    ],
    # ── BT5 Evaluate ── verbs: evaluate, assess, justify ─────────────────────
    # Every pattern contains an explicit judgment criterion.
    # No "suitability for demanding applications" without defining the criterion.
    "BT5": [
        # single-concept with explicit criterion
        "Evaluate the strengths and weaknesses of {A} and justify whether it is the best solution for its purpose.",
        "Evaluate {A} based on relevant criteria such as correctness, efficiency, and ease of use.",
        "Assess the limitations of {A} and explain under what conditions a different structure would be better.",
        "Assess whether the time complexity of {A} justifies its use over simpler alternatives.",
        # two-concept judgment with criterion
        "Evaluate the trade-offs between {A} and {B} for an application with strict time or resource constraints.",
        "Evaluate whether {A} or {B} is more appropriate given specific constraints, and justify your answer.",
        "Assess the impact of choosing {A} over {B} and justify which is more appropriate for the given context.",
        "Assess which of {A} or {B} is better suited for solving problems with large input sizes.",
        "Justify why {A} is a better choice than {B} for problems where verification or feasibility is a concern.",
        "Justify why a system designer would choose {A} over {B} when scalability is a priority.",
        # challenge-only patterns
        "Evaluate the impact of {A} on system performance and justify a specific strategy to address it.",
        "Assess the limitations of {A} and propose a specific improvement or alternative.",
        "Justify why resolving {A} is critical to the long-term reliability of {B}.",
    ],
    # ── BT6 Create ── verbs: design, propose, develop ────────────────────────
    # Every pattern specifies what to build AND what problem it must solve.
    # No "real-world application" without a named domain or constraint.
    "BT6": [
        # single-concept with specific deliverable and domain
        "Design a solution using {A} that addresses a real-world problem, explaining your key decisions.",
        "Design an algorithm using {A} to solve a specific computational problem and describe how it works.",
        "Propose a way to use {A} to improve an existing solution, explaining what specific improvement it provides.",
        "Propose an improved version of {A} that addresses its main performance limitation.",
        "Develop a plan to use {A} in a practical system, identifying the key challenges and how to address them.",
        "Develop a guide that helps a developer decide when to use {A} instead of a simpler alternative.",
        # two-concept integration with specific problem
        "Design a system that uses {A} to organise information and {B} to process or retrieve it efficiently.",
        "Design a solution that combines the strengths of {A} and {B} to solve a problem neither can handle alone.",
        "Propose a solution that uses {A} and {B} together to handle both sequential and hierarchical data.",
        "Propose a system that switches between {A} and {B} depending on the size of the dataset.",
        "Develop a model that uses {A} and {B} together to improve the overall efficiency of a complex task.",
        "Develop a prototype that replaces {B} with {A} to improve performance and explain the improvement.",
    ],
}


def load_dataset():
    """Return dict {category: [question_strings]}."""
    data = defaultdict(list)
    try:
        with open(DATASET_PATH, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                cat = row.get("Category", "").strip()
                q   = row.get("Questions", "").strip()
                if cat and q:
                    data[cat].append(q)
    except FileNotFoundError:
        pass
    return dict(data)


# dataset pattern mining removed — too noisy

# =============================================================================
# SECTION 3 — Question generation
# =============================================================================

def _classify_concepts(concepts):
    """
    Split extracted concepts into two semantic buckets:
    - 'system': technologies, methods, structures — usable as subjects of any pattern
    - 'challenge': problems, issues, concerns — only fit in patterns about problems/limitations
    Returns (system_concepts, challenge_concepts)
    """
    _CHALLENGE_WORDS = {
        # nouns that signal a problem/concern
        'issue','issues','problem','problems','concern','concerns',
        'challenge','challenges','limitation','limitations','drawback','drawbacks',
        'vulnerability','vulnerabilities','risk','risks','overhead','bottleneck',
        'consumption','waste','cost','costs','error','errors','failure','failures',
        'attack','attacks','threat','threats','flaw','flaws','gap','gaps',
    }
    system, challenge = [], []
    for c in concepts:
        words = c.lower().split()
        if any(w in _CHALLENGE_WORDS for w in words):
            challenge.append(c)
        else:
            system.append(c)
    return system, challenge


# Single-word concepts that are always meaningful DS structure names
# These are promoted regardless of how many multi-word compounds exist
_DS_STRUCTURE_NAMES = {
    # Primary structure types — sentence subjects, standalone topics
    'array','arrays','stack','stacks','queue','queues',
    'tree','trees','graph','graphs','heap','heaps',
    'linked list','linked lists',
    # Supporting but meaningful standalone concepts
    'pointer','pointers',
}
# Sub-part terms — valid in Pool B but NOT promoted to Pool A
_DS_SUBPART_NAMES = {
    'node','nodes','vertex','vertices','edge','edges',
    'list','lists','set','sets','map','maps',
    'index','indices',
    'pointer','pointers',   # sub-component of linked lists, not a standalone structure
}


# Process/algorithm concepts — valid in Pool B as supporting detail,
# but should NOT be Pool A primary subjects for structure-focused patterns.
# They work as {B} in "how does {A} use {B}" but not as {A} in "define {A}".
_PROCESS_CONCEPTS = {
    # DS / Algorithms
    'graph traversal', 'depth-first search', 'breadth-first search',
    'binary search', 'binary search trees', 'tree traversal',
    'expression evaluation', 'function call management',
    'time complexity', 'space complexity',
    # Architecture
    'pipelining', 'branch prediction', 'out of order execution',
    'instruction fetch', 'instruction decode',
    # Compiler
    'shift reduce', 'leftmost derivation', 'rightmost derivation',
    'constant folding', 'constant propagation', 'strength reduction',
    # ML
    'back propagation', 'forward propagation', 'gradient descent',
    'dimensionality reduction', 'feature extraction',
    # Security
    'sql injection', 'cross site scripting', 'buffer overflow',
    'man in the middle', 'denial of service',
    # IR
    'tf-idf', 'cosine similarity', 'pagerank', 'query expansion',
}


# ─────────────────────────────────────────────────────────────────────────
# Concept category tags — used to enforce valid comparison pairs.
# Concepts in different categories should NOT be compared directly.
# ─────────────────────────────────────────────────────────────────────────

_CAT_STRUCTURE = {
    # Data structures
    'array','arrays','linked list','linked lists','stack','stacks',
    'queue','queues','tree','trees','graph','graphs','heap','heaps',
    'hash table','hash tables','hash map','priority queue',
    'binary tree','binary search tree','avl tree','b tree',
    'adjacency list','adjacency matrix',
}

# OOP-specific categories — kept separate so OOP concepts
# are never compared to data structures or algorithms
_CAT_OOP_PRINCIPLE = {
    # Core OOP principles (cannot be compared to mechanisms or instances)
    'encapsulation','inheritance','polymorphism','abstraction',
    'composition','aggregation','association',
}

_CAT_OOP_MECHANISM = {
    # OOP implementation mechanisms
    'method overriding','method overloading','constructor',
    'getter','setter','access modifier',
    'static method','instance method','virtual method',
}

_CAT_OOP_CONSTRUCT = {
    # OOP language constructs
    'class','object','interface','abstract class','concrete class',
    'subclass','superclass','parent class','base class','derived class',
    'instance','member variable','instance variable',
}

_CAT_ALGORITHM = {
    # Sorting
    'merge sort','quick sort','heap sort','bubble sort','insertion sort',
    'selection sort','radix sort','counting sort',
    # Searching / Graph
    'binary search','linear search','depth-first search','breadth-first search',
    'dijkstra algorithm','kruskal algorithm','prim algorithm',
    'bellman ford','floyd warshall','topological sort',
    # ML algorithms
    'gradient descent','back propagation','k-nearest neighbor',
    'naive bayes','logistic regression',
    # Crypto algorithms
    'rsa algorithm','aes algorithm','des algorithm',
    # Scheduling algorithms
    'round robin','priority scheduling','shortest job first',
    'first come first served','multilevel queue',
    # Page replacement
    'least recently used','optimal page replacement',
    # Compiler algorithms
    'll parser','lr parser','slr parser','lalr parser',
}

_CAT_MECHANISM = {
    # OS mechanisms
    'paging','segmentation','virtual memory','demand paging',
    'page replacement','memory allocation','memory management',
    'process management','process scheduling','process synchronization',
    'deadlock prevention','deadlock avoidance','deadlock detection',
    'mutual exclusion','semaphore','monitor','critical section',
    'context switch','system call','device driver',
    'file system','file allocation','disk scheduling',
    'interprocess communication','shared memory','message queue',
    # Network mechanisms
    'congestion control','flow control','error detection','error correction',
    'routing','packet switching','circuit switching','subnetting',
    'three-way handshake','sliding window',
    # Security mechanisms
    'authentication','authorization','access control','encryption','decryption',
    'firewall','intrusion detection','digital signature',
    # DB mechanisms
    'transaction management','concurrency control','query optimization',
    'locking protocol','timestamp ordering',
}

_CAT_RULE = {
    # Probability rules
    'addition rule','multiplication rule','bayes theorem',
    'conditional probability','law of total probability',
    # Math theorems
    'central limit theorem','pythagorean theorem','fundamental theorem',
    'pumping lemma','closure property',
    # CS theorems / laws
    'amdahl law','gustafson law','cap theorem',
    # OOP principles
    'solid principles','solid principle',
}

_CAT_MODEL = {
    # Network models
    'osi model','tcp ip model','tcp model',
    # Software models
    'waterfall model','agile model','spiral model','scrum',
    # ML models
    'neural network','convolutional neural network','recurrent neural network',
    'transformer model','decision tree','random forest','support vector',
    # DB models
    'relational model','entity relationship','er diagram',
    # Architecture models
    'memory hierarchy','pipeline','instruction set architecture',
}

_CAT_CONCEPT = {
    # Abstract CS concepts
    'inheritance','polymorphism','encapsulation','abstraction',
    'recursion','iteration','modularity','coupling','cohesion',
    # Probability concepts
    'probability','sample space','random variable','expected value',
    'variance','standard deviation','normal distribution',
    'binomial distribution','discrete random variable',
    'continuous random variable','mutually exclusive','independence',
    # Complexity concepts
    'time complexity','space complexity','polynomial time',
    'exponential time','np-complete','np-hard','tractable problem',
    'intractable problem','computational complexity',
    # General concepts
    'cloud computing','virtualization','microservice','containerization',
    'parallel computing','distributed computing','concurrency',
}


def _categorise_concept(c):
    """
    Return the semantic category of a concept string.
    Categories: 'structure', 'algorithm', 'mechanism', 'rule', 'model',
                'oop_principle', 'oop_mechanism', 'oop_construct', 'concept'
    Used to enforce valid comparison pairs — only same-category concepts
    should appear together in comparison patterns.
    """
    cl = c.lower().strip()
    if cl in _CAT_OOP_PRINCIPLE:  return 'oop_principle'
    if cl in _CAT_OOP_MECHANISM:  return 'oop_mechanism'
    if cl in _CAT_OOP_CONSTRUCT:  return 'oop_construct'
    if cl in _CAT_STRUCTURE:      return 'structure'
    if cl in _CAT_ALGORITHM:      return 'algorithm'
    if cl in _CAT_MECHANISM:      return 'mechanism'
    if cl in _CAT_RULE:           return 'rule'
    if cl in _CAT_MODEL:          return 'model'
    if cl in _CAT_CONCEPT:        return 'concept'
    # Heuristic fallbacks
    _ALG_HINTS = ('sort','search','algorithm','traversal','scheduling',
                  'replacement','allocation','detection','prevention','avoidance')
    if any(h in cl for h in _ALG_HINTS):  return 'algorithm'
    _STRUCT_HINTS = ('list','tree','graph','table','queue','stack','heap','array')
    if any(h in cl for h in _STRUCT_HINTS): return 'structure'
    _MECH_HINTS = ('management','control','protocol','switching','paging',
                   'segmentation','synchronization','communication')
    if any(h in cl for h in _MECH_HINTS): return 'mechanism'
    _RULE_HINTS = ('theorem','rule','law','lemma','principle','axiom')
    if any(h in cl for h in _RULE_HINTS): return 'rule'
    _MODEL_HINTS = ('model','framework','architecture','pattern','paradigm')
    if any(h in cl for h in _MODEL_HINTS): return 'model'
    return 'concept'   # default


# Within oop_construct, only some pairs are genuine alternatives
_OOP_CONSTRUCT_GROUPS = [
    {'class', 'abstract class', 'concrete class', 'interface'},  # type definitions
    {'object', 'instance'},                                        # instances
    {'subclass', 'superclass', 'parent class', 'base class', 'derived class'},  # hierarchy
]

def _same_category(a, b):
    """
    Return True if both concepts are semantically compatible for comparison.
    Uses category matching plus finer compatibility rules within OOP constructs.
    """
    cat_a = _categorise_concept(a)
    cat_b = _categorise_concept(b)
    if cat_a != cat_b:
        return False
    # Within oop_construct, only compare within the same subgroup
    if cat_a == 'oop_construct':
        al, bl = a.lower().strip(), b.lower().strip()
        for grp in _OOP_CONSTRUCT_GROUPS:
            if al in grp and bl in grp:
                return True
        return False   # different subgroups — not comparable
    return True


def _build_concept_pool(concepts, text=""):
    """
    Build two ordered pools for {A} and {B} slots.

    Pool A — PRIMARY concepts (main question subjects):
      1. Sentence-subject concepts (appear at start of a sentence in the text)
      2. DS structure names (array, linked list, stack, queue, tree, graph, etc.)
      3. Other multi-word system concepts
    Pool B — SECONDARY concepts (detail / attribute / support):
      Everything in Pool A, plus attribute-level compounds and single words.

    Challenge concepts are separated for challenge-aware patterns.
    """
    system, challenge = _classify_concepts(concepts)

    # ── Step 1: Detect sentence-subject concepts ────────────────────────────
    # Concepts that appear at the start of a sentence are the TEXT'S main topics.
    import re as _re
    sentences = _re.split(r'(?<=[.!?])\s+', text.strip()) if text else []
    sentence_starts = [s.strip().lower()[:60] for s in sentences]

    def _is_sentence_subject(c):
        cl = c.lower()
        # Match both direct ("Arrays store...") and article-prefixed
        # ("A deterministic algorithm...", "The Clique problem...")
        return any(
            ss.startswith(cl) or
            ss.startswith('a ' + cl) or
            ss.startswith('an ' + cl) or
            ss.startswith('the ' + cl)
            for ss in sentence_starts
        )

    # ── Step 2: Classify each system concept by priority tier ───────────────
    tier1 = []  # sentence subjects — always in Pool A
    tier2 = []  # DS structure names or short known-term multi-word concepts
    tier3 = []  # other multi-word attribute compounds (contiguous memory, etc.)
    tier4 = []  # single-word tech terms that aren't structure names

    for c in system:
        words = c.split()
        is_subj  = _is_sentence_subject(c)
        is_ds    = c.lower() in _DS_STRUCTURE_NAMES or (
                   len(words) == 1 and c.lower() in _DS_STRUCTURE_NAMES)
        is_multi = len(words) >= 2

        is_subpart  = c.lower() in _DS_SUBPART_NAMES
        is_process  = c.lower() in _PROCESS_CONCEPTS
        if is_subj and not is_subpart and not is_process:
            tier1.append(c)
        elif is_ds and not is_subpart and not is_process:
            tier2.append(c)
        elif is_multi:
            tier3.append(c)
        else:
            tier4.append(c)

    # Deduplicate DS names: prefer plural form
    _seen_ds_root = {}
    for c in tier1 + tier2:
        key = c.lower().rstrip('s') if c.lower().endswith('s') else c.lower()
        if key not in _seen_ds_root or len(c) > len(_seen_ds_root[key]):
            _seen_ds_root[key] = c
    tier1_dedup = []
    tier2_dedup = []
    _used = set()
    for c in tier1:
        key = c.lower().rstrip('s') if c.lower().endswith('s') else c.lower()
        canonical = _seen_ds_root.get(key, c)
        if canonical not in _used:
            tier1_dedup.append(canonical)
            _used.add(canonical)
    for c in tier2:
        key = c.lower().rstrip('s') if c.lower().endswith('s') else c.lower()
        canonical = _seen_ds_root.get(key, c)
        if canonical not in _used:
            tier2_dedup.append(canonical)
            _used.add(canonical)

    # ── Step 3: Assemble pools ───────────────────────────────────────────────
    # Pool A: primary concepts only — sentence subjects + DS structure names
    pool_a = tier1_dedup + tier2_dedup
    # If still too small, add tier3 multi-word compounds
    if len(pool_a) < 3:
        pool_a = pool_a + [c for c in tier3 if c not in pool_a]

    # Pool B: everything — Pool A + attribute compounds + single-word terms
    # (B provides the 'detail' side of two-concept questions)
    pool_b_extra = [c for c in tier3 + tier4 if c not in pool_a]
    pool_b = pool_a + pool_b_extra

    if not pool_a:
        pool_a = pool_b = ["the system", "the process", "the model", "the network"]

    return pool_a, pool_b, challenge


def _is_structural_pattern(p):
    """Return True if this pattern asks about internal structure/design.
    These patterns only make sense for data structures and algorithms,
    NOT for mathematical rules, theorems, or abstract concepts.
    """
    _STRUCTURAL_KEYWORDS = {
        'internal design', 'internal structure', 'organised internally',
        'how each part supports', 'insertion and deletion',
        'time complexity of the key operations',
        'time complexity, memory usage',
        'memory usage and access',
    }
    pl = p.lower()
    return any(kw in pl for kw in _STRUCTURAL_KEYWORDS)


def _is_comparison_pattern(p):
    """Return True if this pattern directly compares {A} and {B}.
    Comparison patterns must pair same-tier concepts (both structures,
    or both metrics) to avoid nonsensical pairings like 'arrays vs space complexity'.
    """
    pl = p.lower()
    _COMPARISON_MARKERS = {
        'over {b}', 'versus {b}', '{a} and {b}', '{a} or {b}',
        'between {a} and', '{a} vs',
        'better choice than {b}', 'rather than {b}',
        'complements {b}', 'more efficient than {b}',
        'performs better than {b}', 'outperforms {b}', 'better than {b}',
        'currently using {b}', 'improve the search efficiency of a system currently using {b}',
        '{a} or {b}', 'chosen over {b}', 'instead of {b}',
        '{a} for storage and {b}', '{a} for indexing and {b}',
        'fast access of {a}', 'switches between {a} and {b}',
        # BT1 two-concept patterns — also need same-category enforcement
        'difference between {a} and {b}', '{a} is used instead of {b}',
        'relationship between {a} and {b}', '{a} and {b} in oop',
        '{a} is preferred over {b}',
    }
    return any(kw in pl for kw in _COMPARISON_MARKERS)


def _pick_concepts_for_question(pool_a, pool_b, n_slots, q_index,
                                usage_counter=None, relationships=None, **kwargs):
    """
    Pick concepts for one question.
    - {A} from pool_a (multi-word preferred), choosing least-used concept
    - {B} from pool_b, preferring concepts that co-occur with A in the text
    - relationships: list of (A, B, count) from extract_relationships
    - usage_counter: shared Counter across all questions to track concept reuse
    """
    if usage_counter is None:
        usage_counter = Counter()
    if relationships is None:
        relationships = []

    # Build a lookup: given A, which B concepts are text-related?
    related = {}
    for ra, rb, _ in relationships:
        related.setdefault(ra.lower(), []).append(rb)
        related.setdefault(rb.lower(), []).append(ra)

    # Pick A: least-used in pool_a, with q_index as tiebreaker
    def _usage_key_a(c):
        return (usage_counter[c], pool_a.index(c))
    a = min(pool_a, key=_usage_key_a)

    result = [a]
    if n_slots >= 2:
        # Prefer B concepts that are text-related to A and least used
        text_related_b = [c for c in related.get(a.lower(), [])
                          if c in pool_b and c != a and not _too_similar(a, c)]
        other_b        = [c for c in pool_b
                          if c != a and not _too_similar(a, c)
                          and c not in text_related_b]

        # For comparison patterns, enforce same semantic CATEGORY
        # (structure vs structure, algorithm vs algorithm, etc.)
        # This prevents "Dijkstra vs linked list" or "queue vs probability" pairings
        pattern = kwargs.get('pattern', '')
        if _is_comparison_pattern(pattern):
            # Prefer: same category AND text-related
            same_cat = [c for c in pool_a if c != a
                        and not _too_similar(a, c)
                        and _same_category(a, c)]
            if same_cat:
                text_related_b = [c for c in same_cat if c in text_related_b] or same_cat
                other_b        = [c for c in same_cat if c not in text_related_b]
            else:
                # Fallback: same tier (pool_a) even if different category
                same_tier = [c for c in pool_a if c != a and not _too_similar(a, c)]
                text_related_b = [c for c in same_tier if c in text_related_b] or same_tier
                other_b        = [c for c in same_tier if c not in text_related_b]

        def _usage_key_b(c):
            return (usage_counter[c], pool_b.index(c) if c in pool_b else 999)

        # Try text-related first, fall back to any available
        candidates = text_related_b + other_b
        if not candidates:
            candidates = [c for c in pool_b if c != a]
        b = min(candidates, key=_usage_key_b) if candidates else a
        result.append(b)

    return result


def _too_similar(a, b):
    """Return True if a and b are too similar to pair in the same question."""
    a_l, b_l = a.lower(), b.lower()
    if a_l == b_l or a_l in b_l or b_l in a_l:
        return True
    # Reject if they share a non-trivial content word (len > 4)
    _TINY = {'data','with','from','into','that','this','over','self','high','main'}
    a_words = {w for w in a_l.split() if len(w) > 4 and w not in _TINY}
    b_words = {w for w in b_l.split() if len(w) > 4 and w not in _TINY}
    if a_words & b_words:   # shared significant word → too similar
        return True
    return False


def _fill_pattern(pattern, pool_a, pool_b, q_index, usage_counter=None, relationships=None):
    """Fill {A} and {B} slots using usage-aware, relationship-biased concept pools."""
    if usage_counter is None:
        usage_counter = Counter()

    slots = sorted(set(re.findall(r'\{[A-C]\}', pattern)))
    picks = _pick_concepts_for_question(pool_a, pool_b, len(slots),
                                        q_index, usage_counter, relationships,
                                        pattern=pattern)

    # Guard: reject picks that (a) appear verbatim in template,
    #        (b) are already used in this question, or
    #        (c) are a substring of another pick
    template_lower = pattern.lower()
    all_pool = pool_a + [c for c in pool_b if c not in pool_a]
    guarded   = []
    used_in_q = set()

    for i, pick in enumerate(picks):
        def _bad(candidate):
            if candidate.lower() in template_lower:
                return True
            if candidate in used_in_q:
                return True
            for chosen in guarded:
                if _too_similar(candidate, chosen):
                    return True
            return False

        if _bad(pick):
            # Try least-used alternative first
            alts = sorted(
                [c for c in all_pool if not _bad(c)],
                key=lambda c: (usage_counter[c], all_pool.index(c))
            )
            pick = alts[0] if alts else pick

        guarded.append(pick)
        used_in_q.add(pick)
        usage_counter[pick] += 1   # record usage

    q = pattern
    for slot, val in zip(slots, guarded):
        q = q.replace(slot, val, 1)
    q = q.strip()

    # Grammar fix: plural concepts need plural verbs
    # e.g. "voting systems works" → "voting systems work"
    # "scalability issues is" → "scalability issues are"
    _PLURAL_SUFFIXES = ('systems', 'issues', 'concerns', 'challenges',
                        'limitations', 'applications', 'features',
                        'methods', 'approaches', 'techniques', 'lists',
                        'arrays', 'trees', 'graphs', 'elements', 'records')
    for concept in guarded:
        if any(concept.lower().endswith(s) for s in _PLURAL_SUFFIXES):
            # Fix singular verb forms that follow the concept directly
            q = re.sub(r'\b' + re.escape(concept) + r' works\b',
                       concept + ' work', q, flags=re.I)
            q = re.sub(r'\b' + re.escape(concept) + r' is\b',
                       concept + ' are', q, flags=re.I)
            q = re.sub(r'\b' + re.escape(concept) + r' has\b',
                       concept + ' have', q, flags=re.I)
            q = re.sub(r'\b' + re.escape(concept) + r" doesn't\b",
                       concept + " don't", q, flags=re.I)

    if q and q[-1] not in '.?!':
        q += '.'
    return q[0].upper() + q[1:] if q else q


def _detect_domain(pool_a, pool_b):
    """
    Detect the subject domain from the concept pools.
    Returns one of: 'cs_technical', 'mathematics', 'mixed'

    'cs_technical' — data structures, algorithms, OS, networks, etc.
    'mathematics'  — probability, statistics, calculus, discrete math
    'mixed'        — contains both or unrecognised
    """
    _OOP_INDICATORS = {
        'encapsulation','inheritance','polymorphism','abstraction',
        'composition','interface','method overriding','method overloading',
        'abstract class','subclass','superclass','object oriented',
    }
    all_concepts = [c.lower() for c in pool_a + pool_b]
    oop_score = sum(1 for c in all_concepts if any(m in c for m in _OOP_INDICATORS))
    if oop_score >= 3:
        return 'oop'

    _OOP_INDICATORS = {
        'encapsulation', 'inheritance', 'polymorphism', 'abstraction',
        'class', 'object', 'interface', 'method overriding', 'composition',
        'subclass', 'superclass', 'abstract class',
    }
    oop_score = sum(1 for c in all_concepts if any(m in c for m in _OOP_INDICATORS))
    if oop_score >= 3:
        return 'oop'

    _MATH_INDICATORS = {
        'probability', 'sample space', 'random variable', 'expected value',
        'variance', 'standard deviation', 'normal distribution',
        'binomial distribution', 'addition rule', 'multiplication rule',
        'central limit theorem', 'bayes theorem', 'conditional probability',
        'derivative', 'integral', 'calculus', 'differentiation', 'integration',
        'matrix', 'vector', 'eigenvalue', 'determinant', 'set theory',
        'permutation', 'combination', 'factorial', 'mean', 'median', 'mode',
    }
    _CS_INDICATORS = {
        'array', 'stack', 'queue', 'tree', 'graph', 'linked list',
        'algorithm', 'operating system', 'process', 'memory', 'network',
        'database', 'compiler', 'encryption', 'class', 'object',
        'cpu', 'cache', 'pipeline', 'thread', 'deadlock',
    }
    math_score = sum(1 for c in all_concepts if any(m in c for m in _MATH_INDICATORS))
    cs_score   = sum(1 for c in all_concepts if any(m in c for m in _CS_INDICATORS))

    if math_score > cs_score and math_score >= 2:
        return 'mathematics'
    if cs_score > math_score and cs_score >= 2:
        return 'cs_technical'
    return 'mixed'


# =============================================================================
# Domain-specific pattern overrides
# =============================================================================
_OOP_PATTERN_OVERRIDES = {
    # ── BT1 Remember ── define, list, identify, name ─────────────────────────
    # Only ask for facts directly statable from text — no evaluation needed
    "BT1": [
        "Define {A} and state its role in object-oriented programming.",
        "Define {A} and give one example of how it is used in a class.",
        "List two benefits of using {A} in software design.",
        "List the key characteristics of {A}.",
        "Identify the difference between {A} and {B}.",
        "Identify one situation where {A} is used and describe what it achieves.",
        "Name the OOP principle that {A} belongs to and state what it does.",
        "Name one real-world scenario where {A} is applied and state what it achieves.",
        "State what {A} means and give a short example.",
        "Identify the relationship between {A} and {B} in OOP.",
        "List the conditions under which {A} applies.",
        "Name the four main OOP principles and identify which one {A} represents.",
    ],
    # ── BT2 Understand ── explain, describe, summarize ────────────────────────
    "BT2": [
        "Explain what {A} means and why it is important in OOP design.",
        "Explain the difference between {A} and {B} and when each is used.",
        "Describe how {A} improves the maintainability and reusability of code.",
        "Describe the relationship between {A} and {B} in an OOP system.",
        "Summarize what {A} achieves and what problems it solves.",
        "Summarize the key difference between {A} and {B} in terms of design flexibility.",
        "Explain how {A} supports the principle of hiding implementation details.",
        "Explain why {A} is preferred over {B} in certain design situations.",
        "Describe how {A} and {B} work together to make code more modular.",
        "Describe a real-world analogy that illustrates the concept of {A}.",
        "Summarize how {A} enables code reuse and reduces duplication.",
        "Explain how {A} allows different classes to be used interchangeably.",
    ],
    # ── BT3 Apply ── demonstrate, illustrate, use ─────────────────────────────
    "BT3": [
        "Demonstrate how {A} is implemented by designing a simple class with an example.",
        "Demonstrate how {A} works by writing out a class hierarchy with at least two classes.",
        "Illustrate how {A} is used in a banking system by showing a class with attributes and methods.",
        "Illustrate how {A} and {B} are applied together in the design of a school management system.",
        "Use {A} to design a class that protects its data and only exposes necessary operations.",
        "Use {A} and {B} to show how two different classes can share behaviour while having their own implementations.",
        "Demonstrate how {A} works by designing a class that applies it with a concrete before-and-after example.",
        "Illustrate how {A} enables objects of different types to respond to the same method call differently.",
        "Use {A} to show how two classes with different implementations can be used interchangeably.",
        "Demonstrate the difference between {A} and {B} using a class design example.",
        "Illustrate a design scenario where {A} is more appropriate than {B} and explain why.",
        "Use a step-by-step class design to show how {A} reduces code duplication.",
    ],
    # ── BT4 Analyze ── compare, analyze, examine ──────────────────────────────
    "BT4": [
        "Compare {A} and {B} in terms of code reusability, flexibility, and design complexity.",
        "Compare {A} and {B} and explain which is more appropriate when the relationship between classes may change.",
        "Analyze how {A} and {B} work together to achieve loosely coupled, highly cohesive design.",
        "Analyze the trade-offs of using {A} over {B} in a large-scale software system.",
        "Examine how {A} enforces a specific OOP principle and what design benefits this provides.",
        "Examine the difference between {A} and {B} and explain the scenarios where each is the better design choice.",
        "Analyze why violating {A} can lead to fragile, difficult-to-maintain code.",
        "Analyze how {A} contributes to the extensibility of a software system.",
        "Examine how {A} and {B} differ in how they model the relationship between classes.",
        "Compare the design implications of choosing {A} over {B} for a system with frequently changing requirements.",
        "Analyze the causes and consequences of poor {A} and how they affect {B}.",
        "Examine the factors that determine whether {A} or {B} is the better design choice.",
    ],
    # ── BT5 Evaluate ── evaluate, assess, justify ─────────────────────────────
    "BT5": [
        "Evaluate whether {A} or {B} is the better design choice for a system where requirements change frequently.",
        "Evaluate the impact of {A} on the maintainability and extensibility of a software system.",
        "Assess the limitations of relying too heavily on {A} and explain when a different approach is needed.",
        "Assess the trade-offs between {A} and {B} for a system that must be easy to test and modify.",
        "Justify why {A} is considered a fundamental principle of good OOP design.",
        "Justify why {A} is preferred over {B} when designing systems that must accommodate future extensions.",
        "Evaluate how well {A} supports the goal of writing reusable, modular software.",
        "Assess whether {A} or {B} provides better encapsulation and explain your reasoning.",
        "Evaluate the consequences of not applying {A} in a large object-oriented system.",
        "Justify why {A} is a better design choice than {B} in a system where classes change frequently.",
        "Evaluate the impact of {A} on overall code quality and justify a strategy to apply it effectively.",
        "Assess how {A} affects the testability of a system and suggest specific improvements.",
    ],
    # ── BT6 Create ── design, propose, develop ────────────────────────────────
    "BT6": [
        "Design a class hierarchy for a university system using {A} and {B}, showing at least three classes with their relationships.",
        "Design an OOP solution for a hospital patient record system that applies {A} to protect sensitive data.",
        "Propose a class design for an e-commerce system that uses {A} to allow different payment methods without changing existing code.",
        "Propose an OOP design that uses {A} and {B} to model a vehicle fleet system with different vehicle types.",
        "Develop a class diagram for a library system that demonstrates {A}, showing which classes inherit from which.",
        "Develop an object-oriented model for a notification system using {A} to send messages through multiple channels.",
        "Design a solution that uses {A} to allow a system to work with new object types without modifying existing code.",
        "Design a class structure that uses both {A} and {B} and explain why you chose each in your design.",
        "Propose an improvement to an existing design that violates {A} by refactoring it to apply the principle correctly.",
        "Propose a design that uses {A} to decouple components so that each part can be developed and tested independently.",
        "Develop a prototype that demonstrates how {A} enables a system to be extended with new functionality.",
        "Develop a model that uses {A} and {B} together to improve the modularity of a complex software system.",
    ],
}


_MATH_PATTERN_OVERRIDES = {
    "BT1": [
        "Define {A} and state what it represents.",
        "Define {A} and give one example of how it is calculated.",
        "List the conditions under which {A} applies.",
        "List the key properties of {A}.",
        "Identify the difference between {A} and {B}.",
        "Identify when {A} is used instead of {B}.",
        "Name the formula used to calculate {A}.",
        "Name one real-world situation where {A} is relevant.",
        "State the rule or theorem that defines {A}.",
        "State the relationship between {A} and {B} as defined in the text.",
        "Identify what {A} measures and what values it can take.",
        "Name the assumption required for {A} to hold.",
    ],
    "BT2": [
        "Explain what {A} means and how it is interpreted.",
        "Explain the difference between {A} and {B} and when each applies.",
        "Describe how {A} is calculated and what each component represents.",
        "Describe the relationship between {A} and {B}.",
        "Summarize what {A} tells us and what its limitations are.",
        "Summarize the conditions under which {A} and {B} give the same result.",
        "Explain why {A} is important in understanding probability and uncertainty.",
        "Explain how {A} changes when {B} changes.",
        "Describe a situation where {A} would be applied and what it would tell you.",
        "Describe the difference between {A} and {B} using a simple example.",
        "Summarize the key assumptions required for {A} to be valid.",
        "Explain how {A} is related to {B} mathematically.",
    ],
    "BT3": [
        "Calculate {A} for a specific example and show each step of your working.",
        "Use {A} to solve a numerical problem, showing all steps clearly.",
        "Demonstrate how {A} is applied by working through a specific example with given values.",
        "Illustrate how {A} works using a probability tree or table.",
        "Apply {A} to find the probability of a specific event occurring.",
        "Use {A} and {B} together to solve a two-step probability problem.",
        "Demonstrate the difference between {A} and {B} by applying each to the same problem.",
        "Calculate the result when {A} is applied to a dataset of five values.",
        "Illustrate how {A} produces a different result from {B} with a worked example.",
        "Use {A} to determine the likelihood of an outcome in a given experiment.",
        "Apply {A} step by step to verify whether two events satisfy the condition for {B}.",
        "Show how {A} is computed when the sample space has six equally likely outcomes.",
    ],
    "BT4": [
        "Compare {A} and {B} in terms of when each is applicable and what each measures.",
        "Analyze the conditions under which {A} holds and explain what happens when they are violated.",
        "Examine the relationship between {A} and {B} and explain how one affects the other.",
        "Analyze why {A} produces a different result from {B} in the same scenario.",
        "Compare the assumptions required for {A} and {B} and explain the consequences of violating them.",
        "Examine how {A} behaves as the number of trials or sample size increases.",
        "Analyze the effect of changing a parameter in {A} on the overall outcome.",
        "Compare {A} and {B} using a specific numerical example to show the difference.",
        "Examine the mathematical relationship between {A} and {B}.",
        "Analyze when {A} provides a good approximation for {B} and when it does not.",
        "Analyze the causes and consequences of {A} and their impact on {B}.",
        "Examine the main factors that affect {A} and how they limit {B}.",
    ],
    "BT5": [
        "Evaluate whether {A} is an appropriate model for a given real-world situation and justify your reasoning.",
        "Assess the limitations of {A} and explain in what circumstances a different approach would be needed.",
        "Justify why {A} is used in preference to {B} for a specific type of problem.",
        "Evaluate the trade-offs between using {A} and {B} and determine which is more appropriate.",
        "Assess whether the assumptions of {A} are satisfied in a given scenario and what the implications are.",
        "Justify why {A} is considered a more powerful or general result than {B}.",
        "Evaluate whether {A} or {B} provides a better description of a given dataset.",
        "Assess the impact of violating the assumptions of {A} on the validity of your conclusions.",
        "Justify why {A} and {B} must both be considered when solving a given type of problem.",
        "Evaluate the significance of {A} in the context of real-world decision making.",
        "Evaluate the impact of {A} on the accuracy of results and justify a strategy to address it.",
        "Assess how {A} affects the reliability of conclusions drawn from {B}.",
    ],
    "BT6": [
        "Design a probability model for a real-world scenario using {A} and calculate a specific outcome.",
        "Propose a method to determine whether {A} or {B} is the appropriate model for a given experiment.",
        "Develop a step-by-step procedure to calculate {A} for a new type of problem not covered in the text.",
        "Design an experiment that demonstrates the difference between {A} and {B} using real or simulated data.",
        "Propose a solution that uses {A} and {B} together to answer a multi-step probability question.",
        "Develop a plan to verify whether a real-world dataset follows {A} and explain your reasoning.",
        "Design a method to estimate {A} from observed data and explain how you would validate your estimate.",
        "Propose an extension of {A} that addresses one of its key limitations.",
        "Develop a model that uses {A} to predict the outcome of a process involving uncertainty.",
        "Design a simulation that uses {A} and {B} to demonstrate a key probability result.",
        "Develop a model that combines {A} and {B} to solve a complex probabilistic problem.",
        "Propose a strategy that uses {A} to improve the accuracy of predictions based on {B}.",
    ],
}


def generate_questions_from_input(text, counts):
    """
    Generate Bloom's Taxonomy questions from user input text.

    Args:
        text   : str  — user's topic passage
        counts : dict — {level_name: int}

    Returns:
        dict {level_name: [question_str, ...]}
    """
    concepts                  = extract_concepts(text)
    pool_a, pool_b, challenge = _build_concept_pool(concepts, text)

    # Detect domain and select appropriate pattern set
    domain = _detect_domain(pool_a, pool_b)
    active_patterns = (
        _MATH_PATTERN_OVERRIDES if domain == 'mathematics'
        else _OOP_PATTERN_OVERRIDES  if domain == 'oop'
        else _STRUCTURAL_PATTERNS
    )

    # Extract text-grounded concept relationships to bias pairing
    # Pairs that co-occur in the same sentence get pre-seeded usage advantage
    relationships = extract_relationships(text, pool_a + pool_b)
    # Pre-populate usage counter so least-used picks favour related pairs
    # (concepts appearing in many relationships get slightly higher initial count
    #  so isolated concepts are picked earlier, not dominated by frequent ones)

    # Challenge concepts are merged into pool_b for Analyze/Evaluate only,
    # where patterns like "examine the challenges of {A}" or "assess the limitations
    # of {A}" are semantically correct for problem-type concepts.
    # Strip single-word challenges already covered by a multi-word challenge
    _CHALLENGE_LEVELS = {'BT4', 'BT5'}
    multi_challenge = [c for c in challenge if len(c.split()) >= 2]
    covered_ch = {w for phrase in multi_challenge for w in phrase.split()}
    challenge_clean = [c for c in challenge
                       if len(c.split()) >= 2 or c not in covered_ch]

    results       = {}
    global_q_idx  = 0
    usage_counter = Counter()   # shared across all levels for even distribution

    for level_name, n in counts.items():
        if n <= 0:
            continue
        cat = LEVEL_MAP.get(level_name)
        if not cat:
            continue

        # For analysis/evaluation levels, enrich pool with challenge concepts
        if cat in _CHALLENGE_LEVELS and challenge_clean:
            eff_pool_a = pool_a + [c for c in challenge_clean if c not in pool_a]
            eff_pool_b = pool_b + [c for c in challenge_clean if c not in pool_b]
        else:
            eff_pool_a = pool_a
            eff_pool_b = pool_b

        patterns      = list(active_patterns.get(cat, _STRUCTURAL_PATTERNS.get(cat, [])))
        random.shuffle(patterns)

        # Separate challenge-only patterns (tagged with [CHALLENGE] marker)
        # For these, pool_a must be challenge concepts so the sentence makes sense
        def _is_challenge_pattern(p):
            _CHALLENGE_VERBS_IN_PATTERN = {
                'contribute to', 'causes and consequences', 'can be mitigated',
                'addressing', 'severity of', 'factors that contribute',
                'resolving', 'mitigating', 'severity',
                'justify a strategy', 'strategy to address',
            }
            pl = p.lower()
            return any(kw in pl for kw in _CHALLENGE_VERBS_IN_PATTERN)

        questions     = []
        used_patterns = set()
        used_verbs    = set()

        # First pass: enforce verb diversity (one question per opening verb)
        for pattern in patterns:
            if len(questions) >= n:
                break
            verb = pattern.split()[0].lower()
            if verb in used_verbs:
                continue
            is_challenge_p  = _is_challenge_pattern(pattern)
            is_structural_p = _is_structural_pattern(pattern)

            # Skip challenge-only patterns when text has no challenge concepts
            if is_challenge_p and not challenge_clean:
                continue

            # Skip structural patterns when Pool A contains non-structural concepts
            # (rules, theorems, abstract concepts have no "internal design")
            if is_structural_p:
                pool_categories = {_categorise_concept(c) for c in eff_pool_a}
                has_structural = bool(pool_categories & {'structure','algorithm','mechanism'})
                if not has_structural:
                    continue

            # Route challenge-only patterns to challenge pool if available
            if is_challenge_p and challenge_clean and cat in _CHALLENGE_LEVELS:
                ch_pool = challenge_clean if challenge_clean else eff_pool_a
                q = _fill_pattern(pattern, ch_pool, eff_pool_b,
                                  global_q_idx, usage_counter, relationships)
            else:
                q = _fill_pattern(pattern, eff_pool_a, eff_pool_b,
                                  global_q_idx, usage_counter, relationships)
            questions.append(q)
            used_patterns.add(pattern)
            used_verbs.add(verb)
            global_q_idx += 1

        # Second pass: fill remaining count allowing verb reuse
        for pattern in patterns:
            if len(questions) >= n:
                break
            if pattern not in used_patterns:
                is_challenge_p  = _is_challenge_pattern(pattern)
                is_structural_p = _is_structural_pattern(pattern)

                if is_challenge_p and not challenge_clean:
                    continue

                if is_structural_p:
                    pool_categories = {_categorise_concept(c) for c in eff_pool_a}
                    has_structural = bool(pool_categories & {'structure','algorithm','mechanism'})
                    if not has_structural:
                        continue

                if is_challenge_p and challenge_clean and cat in _CHALLENGE_LEVELS:
                    ch_pool = challenge_clean if challenge_clean else eff_pool_a
                    q = _fill_pattern(pattern, ch_pool, eff_pool_b,
                                      global_q_idx, usage_counter, relationships)
                else:
                    q = _fill_pattern(pattern, eff_pool_a, eff_pool_b,
                                      global_q_idx, usage_counter, relationships)
                questions.append(q)
                used_patterns.add(pattern)
                global_q_idx += 1

        results[level_name] = questions[:n]

    return results


# =============================================================================
# Quick self-test
# =============================================================================
if __name__ == "__main__":
    compiler_text = (
        "A compiler is a software program that translates source code written "
        "in a high-level programming language into machine code that a computer "
        "can execute. The compilation process consists of several phases "
        "including lexical analysis, syntax analysis, semantic analysis, "
        "optimization, and code generation. During lexical analysis, the "
        "compiler converts sequences of characters into tokens. Syntax analysis "
        "checks whether the tokens follow the grammatical rules of the "
        "programming language. Semantic analysis verifies the meaning of "
        "statements and detects errors in the program. Optimization improves "
        "the efficiency of the generated code, while code generation produces "
        "the final machine-level instructions that can be executed by the "
        "processor."
    )

    print("=== Extracted Concepts ===")
    for c in extract_concepts(compiler_text):
        print(" ", c)

    print("\n=== Generated Questions (3 per level) ===")
    counts = {lvl: 3 for lvl in LEVEL_MAP}
    output = generate_questions_from_input(compiler_text, counts)
    q_num  = 1
    for level, qs in output.items():
        print(f"\n{level} Level Questions:")
        for q in qs:
            print(f"  {q_num}. {q}")
            q_num += 1