The current folder contains the following items:
    1- The file 'Stabilo.c' used as main program for HMM learning in [1].
    2- The hearder file 'hmm.h' containing functions and constants required 
	   for the HMM learning process.
    3- The hearder file 'Stabilo.h' containing functions and constants required 
	   for processing the content of statokinegrams using HMMs.
    4- The subdirectory "StatoDirectory" containing examples of test data.
    5- The current helpfile.

The compilation command line of the program 'Stabilo.c' is:

   gcc -Wall -Wl,--stack,536870912 -o Stabilo.exe Stabilo.c

After the compilation, the execution command line:

   Stabilo.exe DirectoryName FreqDiv TimeShift NbSamples MaxIterations Threshold SaveHMMs
   
Where:   
   
    1- 'DirectoryName' is the name of a folder containing the input dataset in the organized format. 
	  The directory "../The_Parkinson_Dataset/Organized/" was used in [1] for this purpose.
      The subdirectory "./StatoDirectory/" already contains data of the subjects 1 and 2 of [2].
	  You can change the content of that subdirectory.
    2- 'FreqDiv' is the parameter SIGMA of down-sampling step explained in Section 4.2 of [1].
    3- 'TimeShift' is the desired time-shift.
    4- 'NbSamples' is the number N used in the Equation 3 of [1] to split the interval [0, 100].
    5- 'MaxIterations' is the maximum number of iterations for the Baum-Welch algorithm.
    6- 'Threshold' is the threshold belonging to [0, 100] and corresponding to the learning rate of the Baum-Welch algorithm.
    7- 'SaveHMMs' is a binary parameter such that (SaveHMMs=0) if we do not want to record the resulting HMMs in a textfile, 
	  otherwise (SaveHMMs=1).
		  
Thus, a typical example of execution command line is: 
      
   Stabilo.exe StatoDirectory 5 1 10 100 100.0 1

This command produces as output the following files and directories:

1- A subdirectory named "Results" under each of the four experimental test 
   condition, for each subject.  The newly created subdirectory "Results" 
   contains the three following files:
        - 'MCs.txt' 
		- 'AllVectors.txt' 
		- 'MeanVector.txt'
		
  As an example, the four following subdirectories will be created for the subject
  whose data are in the subdirectory "./StatoDirectory/1/":
        - "./StatoDirectory/1/Firm/Closed/Results/" 
		- "./StatoDirectory/1/Firm/Open/Results/" 
        - "./StatoDirectory/1/Foam/Closed/Results/" 
		- "./StatoDirectory/1/Foam/Open/Results/" 
   
   a- The file 'MCs.txt' contains the Markov chains produced by the method proposed in [1] (Section 4.2.1).
      These Markov chains are represented as ORDERED sequences of pairs (e,s) such that each pair (e,s) 
      materializes the fact that the symbol (s) is observed from the state (e). These Markov chains are generated 
      following the method explained at the Sections 4.2.1 and 4.2.2 of [1]. Two consecutive pairs (e,s)(x,y) mean that 
	  there is a state transition from the state e to the state x.
	  
   b- The file 'AllVectors.txt' contains the vectors derived from the HMMs as explained in the Section 4.2.4 of [1]. 
       A vector is derived for each file representing a statokinesigram.
	   
   c- The file 'MeanVector.txt' contains the mean vector of the vectors in 'AllVectors.txt' as 
      explained in the Section 4.2.5 and the Equation 10 of [1].
   
2- A file named 'SubjectVectors.txt' containing the final vectors of all the 
   subjects found in the directory "./StatoDirectory/". Each line contains the 
   components of the final vector associated with the subject whose identifier 
   appears in square brackets at the beginning of the line. 

3- If the parameter 'SaveHMMs' is 1, it the program produces a subdirectory 
   named "HMMs" under each of the four experimental test condition, for each subject.  
   The newly created subdirectory "HMMs" contains the following files where 
   $filename$ refers to the name of the file containing the corresponding statokinesigram
   (without its file extension) :
        - 'Lambda_Init_$filename$.bin' 
        - 'Lambda_$filename$.dat' 
		
  As an example, the four following subdirectories will be created for the subject
  whose data are in the subdirectory "./StatoDirectory/1/":
        - "./StatoDirectory/1/Firm/Closed/HMMs/" 
		- "./StatoDirectory/1/Firm/Open/HMMs/" 
        - "./StatoDirectory/1/Foam/Closed/HMMs/" 
		- "./StatoDirectory/1/Foam/Open/HMMs/" 
	  
   a- The file 'Lambda_Init_$filename$.bin' contains the parameters of the initial HMM.
	   
   b- The file 'Lambda_$filename$.dat' contains the parameters of the trained HMM.
________________________________________________________________________________________________   

[1] Denkeng, A. T., Mourad, A. M., Iloga, S., Mba, R. M.,Baazaoui, H., Ndié, T. D., & Romain, O. (2025).
"Efficient Characterization Of The Human BalanceUsing HMMS". 
IEEE Access. Vol. 13, pp. 183456 - 183479, doi: 10.1109/ACCESS.2025.3622375.

[2] Santos, D.A., Duarte, M.
"A public data set of human balance evaluations."
PeerJ 4, 2648 (2016)