This folder contains:
    1- The directory "./Original/" which contains the dataset proposed in [1] in the original format.
       Each statokinesigram is saved in '.txt' file named "PDPDS$X$$MEDICATION$$SURFACE$$EYES$$TRIAL$.txt" where: 
          a- 'X' is the identifier (number) of the subject in his group.
	      b- 'MEDICATION' is either 'on' or 'off'. 
		  c- 'SURFACE' is either 'rs' (firm) or 'us' (foam).
		  d- 'EYES' is either 'ec' (eyes closed) or 'eo' (eyes open).
		  e- 'TRIAL' is the trial number varying between 1 and 6.
		  
	   The following information are available in each statokinesigram::
          a- 'Time[s]': the recording time.
          b- 'GRFap[N]', 'GRFml[N]' and 'GRFv[N]': The forces applied by the subject on the force platform.
		  c- 'Map[N.m]', 'Mml[N.m]', 'Mv[N.m]' and 'Mfree[N.m]': The moments of these forces.
          d- 'COPap[cm]' and 'COPml[cm]': The current coordinates of the Center Of Pressure of the subject.
		  
    2- The directory "./Organized/" which contains the dataset proposed in [1] in an organized format.
	   The original data are grouped in this order by subject identifier, surface and test condition. 
       As an example, the subdirectory "./Organized/1/Firm/Closed/" contains all the statokinesigrams 
	   recorded for the subject number 1, on a firm surface, under the test condition 'Eyes Closed'. 
	   Similarly, the subdirectory "./Organized/Foam/1/Open/" contains all the statokinesigrams recorded 
	   for the subject number 1, on a foam surface, under the test condition 'Eyes Open'.
	   
	3- The file 'PDPDSinfo.txt' containing detailed insights related to each statokinesigram.  
________________________________________________________________________________________________   

[1] Oliveira, C.E.N., Souza, C., Treza, R.d.C., Hondo, S.M., Los Angeles, E., Bernardo, C., Shida, T.K.F., Oliveira, L., Novaes, T.M., Campos, D.d.S.F., et al.
"A public data set with ground reaction forces of human balance in individuals with parkinson’s disease." 
Frontiers in neuroscience 16, 865882 (2022)
