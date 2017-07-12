# lam
Latent Argument Model

# Requirements

Python 3

Numpy

NLTK

[stop-words](https://pypi.python.org/pypi/stop-words)

[Palmetto](http://139.18.2.164/mroeder/palmetto/palmetto-0.1.0-jar-with-dependencies.jar)

[Wikipedia_db](http://139.18.2.164/mroeder/palmetto/Wikipedia_bd.zip)

# Structure


    lam.py (class implementing the latent argument model)

    explore_lam.py (a script to explore different LAM configurations based on the content of configuration.xml)
    configuration.yml
    
    |_data
      |_ mpqa_subjectivity_priors.txt
      
    |_model_init
      |_ subjective_lexicon.py (functions to convert the MPQA subjective lexicon into
        a file of subjectivity priors)
        
    |_eval
    
    hcd.tsv (The House of Common Debate corpus)
    
# configuration.xml

	PATH_DOCS: Docs to process
	PATH_SUBJECTIVITY_DATA: Path to the file containing the subjectivity priors
	PATH_OUTPUT: Path to the directory where the output will be stored
	TOPICS: Number of topics to explore (separated by commas)
	VIEWPOINTS: Number of viewpoints per topic (separated by commas)
	X: switch strategy lex_pos|pos|lex|random|cpt
	|_lex_pos: Uses subjectivity priors and postags to assign an initial type to each word.
	|_lex: Uses subjectivity priors to assign an initial type to each word.
	|_pos: Uses part-of-speech tags to assign an initial type to each word.
	|_random: Each word is initially assigned a type randomly.
	|_cpt: The way the sampling is done is also different (topics and viewpoints are sampled separately)
	SENTENCE_EXTRACTION: discriminative|generative
    WORD_ASSIGNMENT: regular
	RUNNING_METHOD: topics_first
	HIGH_VIEWPOINT_HIGH_TOPIC: true|false. True if a sentence only can be top perspective sentence iff it is 
                               representative of the corresponding topic more than any otherone else.
    REMOVE_AMBIGUOUS_VIEWPOINT_SENTENCES: true|false. True for not to considering top perspective sentences those
                                          occurring at more than one perspective.
	PATH_PALMETTO_PYSCRIPT
	PATH_PALMETTO_JAR: Download from: http://139.18.2.164/mroeder/palmetto/palmetto-0.1.0-jar-with-dependencies.jar
    PATH_WIKIPEDIA_DUMP: Download from http://139.18.2.164/mroeder/palmetto/Wikipedia_bd.zip

# Execution

python explore_lam.py configuration.yml

# Reference

David Vilares and Yulan He, Detecting Perspectives in Political Debates. In the Proceedings of The 14th International Conference on Empirical Methods on Natural Language Processing (EMNLP 2017), Copenhagen, Denmark, Sep. 2017.
