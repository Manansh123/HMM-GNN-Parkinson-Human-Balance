This folder contains the results of different executions of the method proposed in [1] with various time-shifts(1, 5 and 10) 
It contains the subdirectories "./Shift_$KAPPA$/", with (KAPPA = 1, 5, 10). 
Each subdirectory has the following items:
   1- The subdirectory "./Shift_$KAPPA$/Organized_Processed/" which contains the dataset proposed in [2] in an organized format 
     alongside with the corresponding experimental results.
   2- The file './Shift_$KAPPA$/SubjectVectors.txt' containing the final vectors of all the subjects in the dataset proposed in [2] after the method described in [1] 
     has been applied on it with the corresponding time-shift. 
   3- The file './Shift_$KAPPA$/Vectors.txt' which is a copy of the file './Shift_$KAPPA$/SubjectVectors.txt' where each line 
      starts with the subjects' Id WITHOUT squared bracked, and the lines of this file are sorted in increasing NUMERIC order 
	  of the subjects' identifiers. The file './Shift_$KAPPA$/Vectors.txt' must be MANUALLY generated from the content of the file "SubjectVectors.txt". 
________________________________________________________________________________________________   

[1] Denkeng, A. T., Mourad, A. M., Iloga, S., Mba, R. M.,Baazaoui, H., Ndié, T. D., & Romain, O. (2025).
"Efficient Characterization Of The Human BalanceUsing HMMS". 
IEEE Access. Vol. 13, pp. 183456 - 183479, doi: 10.1109/ACCESS.2025.3622375.

[2] Santos, D.A., Duarte, M.
"A public data set of human balance evaluations."
PeerJ 4, 2648 (2016)