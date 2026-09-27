This folder contains:
    1- The directory "./Original/" which contains the dataset proposed in [1] in the original format.
       Each statokinesigram is saved in '.txt' file named 'BDS$NUMBER$.txt' where NUMBER is the 
	   number of the statokinesigram in the dataset. The following information are available in 
	   each statokinesigram:
          a- 'Time[s]': the recording time.
          b- 'Fx[N]', 'Fy[N]' and 'Fz[N]': The forces applied by the subject on the force platform.
		  c- 'Mx[N.m]', 'My[N.m]' and 'Mz[N.m]': The moments of these forces.
          d- 'COPx[cm]' and 'COPy[cm]': The current coordinates of the Center Of Pressure of the subject.
		  
    2- The directory "./Organized/" which contains the dataset proposed in [1] in an organized format.
	   The original data are grouped in this order by subject identifier, surface and test condition. 
       As an example, the subdirectory "./Organized/1/Firm/Closed/" contains all the statokinesigrams 
	   recorded for the subject number 1, on a firm surface, under the test condition 'Eyes Closed'. 
	   Similarly, the subdirectory "./Organized/Foam/1/Open/" contains all the statokinesigrams recorded 
	   for the subject number 1, on a foam surface, under the test condition 'Eyes Open'.
	   
	3- The files 'OrganizeDataset.c' and 'FilesList.csv' used for formating the original dataset in the 
	   organized format as follow:   
________________________________________________________________________________________________   

The compilation command line of the program 'OrganizeDataset.c' is:

    gcc -Wall -Wl,--stack,536870912 -o OrganizeDataset.exe OrganizeDataset.c

After the compilation, the execution command line:

    OrganizeDataset.exe FilesList OrganizedDirectory OriginalDirectory
   
Where:   

     - 'FilesList' is a file containing the list of all the statokinesigram file names appearing in the dataset, 
	   each statokinesigram been attached with its corresponding subject identifier, its surface and its eye condition.
	   The file 'FilesList.csv' was used in [2] for this purpose.
	   
	 - 'OrganizedDirectory' is the name of the organized directory produced as output. 
	   This directory is created by the program, DO NOT CREATE IT YOURSELF.
	   
	 - 'OriginalDirectory' is the name of the directory containing the original dataset.
		  
Thus, a typical example of execution command line is: 
               
    OrganizeDataset.exe FilesList.csv Organized Original
________________________________________________________________________________________________   

[1] Santos, D.A., Duarte, M.
"A public data set of human balance evaluations." 
PeerJ 4, 2648 (2016)

[2] Denkeng, A. T., Mourad, A. M., Iloga, S., Mba, R. M.,Baazaoui, H., Ndié, T. D., & Romain, O. (2025).
"Efficient Characterization Of The Human BalanceUsing HMMS". 
IEEE Access. Vol. 13, pp. 183456 - 183479, doi: 10.1109/ACCESS.2025.3622375.
