# language_matching_perplexity_analysis
Language matching system using n-grams and comparison of kneser-ney smoothing vs add-k smoothing vs no smoothing

Based on one of the components of the excellent work done by Prof. Grzegorz Kondrak and Bradley Hauer
Uses perplexity values for finding most similar language from a multi-language corpus
Tested using the Universal declaration of Human rights

PLEASE READ "4. Execution" BEFORE RUNNING PROGRAM (langid.py) 
---------------------------------------------------------------------------------------------------------------------------------------------------------------
TABLE OF CONTENTS:
____________________
1. Software requirements
2. Installation
3. Implementation
4. Execution
5. Results


----------------------------------------------------------------------------------------------------------------------------------------------------------------
1. *** SOFTWARE REQUIREMENTS ***
-> 	python 2.7.13
-> 	nltk 3.2.3
-> 	Microsoft windows (for command line argument parsing, not tested for other operating systems)


----------------------------------------------------------------------------------------------------------------------------------------------------------------
2. *** INSTALLATION ***
-> 	Install python version 2.7 or higher 
-> 	Install nltk version 3.2 or higher
		> pip install -U nltk
-> 	Check nltk installation by importing nltk in python environment
-> 	Download all nltk packages:
  		python> import nltk
  		python> nltk.download()


-----------------------------------------------------------------------------------------------------------------------------------------------------------------
3. *** IMPLEMENTATION ***

->      Tuning the value of 'n' for unsmoothed, Kneser-Ney Smoothing and Laplace Add One Smoothing techniques:
	______________________________________________________________________________________________________

	On executing the system over the dev set for different values of 'n'<7, we get the following number of accurate matches:
	- Unsmoothed:
		# n=1 : 9/55 **
		# n=2 : 0/55
		# n=3 : 0/55
		# n=4 : 0/55
		# n=5 : 0/55
		# n=6 : 0/55
	- Laplace Add-one:
		# n=1 : 48/55
		# n=2 : 52/55
		# n=3 : 53/55 **
		# n=4 : 52/55
		# n=5 : 50/55
		# n=6 : 50/55
	- Kneser-ney:
		# n=3 : 53/55 **	Kneser-ney has been executed for only n=3, since Kneser-ney smoothing implementation in NLTK supports only trigrams
		
	** denotes value chosen for final modelling.
	
	Therefore, the values of 'n' are chosen such that the accuracy is highest.
		- For Unsmoothed: n = 1 
		- For Laplace Add One Smoothing: n = 3
		- For Kneser-Ney Smoothing: n = 3
	Values of n>=7 have not been considered because it yields no improvement in accuracy, while making computing time much higher. 
  Users are welcome to try and improve this.
	
	
------------------------------------------------------------------------------------------------------------------------------------------------------------------
4. *** EXECUTION ***
->	Paste training, development and test files into the extracted directory with folders named "811_a1_train", "811_a1_dev" and "811_a1_test_final" respectively
->	Ensure that all training, development and test files follow naming scheme given in data sets given below:
		- "udhr-*.txt.tra" for training files
		- "udhr-*.txt.dev" for dev set files
		- "udhr-*.txt.tes" for test set files
->	If dev files are to be tested, uncomment line 306
->	If final test files are to be tested, uncomment line 309
->	If both dev and final test files are to be tested, uncomment both lines 306 and 309
-> 	Running the program:
	______________________________________________________________________________________________________
		
	- Execute the program from command line with the following arguments as parameters to select which type of smoothing is needed:
		> Unsmoothed: --unsmoothed
		> Kneser-Ney: --kneser-ney
		> Laplace Add One Smoothing: --laplace

	- Execute using (example for kneser-ney smoothing):
		pwd $ python langid.py --kneser-ney
		
	- The results are stored in plain text files in the following format:
		udhr-01.txt.tes   udhr-kri.txt.tra   6.8143054501   3
		udhr-02.txt.tes   udhr-blu.txt.tra   5.66133490772   3
		udhr-03.txt.tes   udhr-cjk_AO.txt.tra   5.16606415135   3
		udhr-04.txt.tes   udhr-deu_1996.txt.tra   8.13813212533   3
		
		> First value is the name of file being tested
		> Second value is name of guess from system
		> Third value is perplexity value
		> Fourth value is length of n-gram chosen for given model
-> 	################################IMPORTANT#################################
	When executing the program for one language model (ex. unsmoothed), other result files will be purged and emptied,
  kindly ensure backing up of result files if needed for documentation


---------------------------------------------------------------------------------------------------------------------------------------------------------------------
5. *** RESULTS ***
->	Unsmoothed language models give least accuracy over dev files, as expected with a maximum accuracy of 9/55 when using n=1
->	Laplace add-one smoothed language models give varying accuracy over dev files, with a maximum accuracy of 53/55 correct guesses when using n=3
->	Kneser-Ney smoothed language models give accuracy of 53/55 when using n=3, however this accuracy cannot be checked for different values of n, since nltk
implementation of Kneser-Ney smoothing works only for trigrams. Accuracy is expected to be higher for higher values of n,
until a point of diminishing returns is reached, which could be tested in the future using self-implemented Kneser-Ney smoothing
