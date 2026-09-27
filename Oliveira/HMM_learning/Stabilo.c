#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include <time.h>
#include <dirent.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "Stabilo.h"

/*
____________________________________________________________________________________________   
   Compilation command line:

   gcc -Wall -Wl,--stack,536870912 -o Stabilo.exe Stabilo.c
________________________________________________________________________________________________   
   Execution command line:

   Stabilo.exe DirectoryName FreqDiv TimeShift NbSamples MaxIterations Threshold SaveHMMs
________________________________________________________________________________________________   
   Example: 
   
   Stabilo.exe StatoDirectory 5 1 10 100 100.0 1
________________________________________________________________________________________________   
*/

int main(int NbArg, char ** MesArg) {
	Stato S;
	double MaxAngle, MaxDist, Vect[MaxSymbols+2], MeanVect[MaxSymbols+2];
	double epsilon,Threshold;
	int i,j,k,l,m,FreqDiv, TimeShift, NbSamples, NbStatos, NbDirectories, MaxIterations,SaveHMMs;
	HMM Lambda_Init, Lambda;
	FILE *f, *g, *h, *z;
	char HMM_DirectoryName[100], HMM_FileName[100], FullFileNames[MaxFilesInDirectory][100];
	char MCs_FileName[100],Vectors_FileName[100],Results_DirectoryName[100], FileNames[MaxFilesInDirectory][50];
	char Current_DirectoryName[100], FullDirectoryNames[MaxFilesInDirectory][100], DirectoryNames[MaxFilesInDirectory][50];
	char Final_Vectors_FileName[100], Mean_Vector_FileName[100];
	char Configurations[4][50] = {"\\Firm\\Open","\\Firm\\Closed","\\Foam\\Open","\\Foam\\Closed"};
	
	clock_t overall_t;
	double Reading_time, HMM_time, Vector_time;
	
	FreqDiv = atoi(MesArg[2]);
	TimeShift = atoi(MesArg[3]);
	NbSamples = atoi(MesArg[4]);
	MaxIterations = atoi(MesArg[5]);
	Threshold = atof(MesArg[6]);
	SaveHMMs = atoi(MesArg[7]);
	epsilon = 1e-20;
		
	Reading_time = 0.0;
	HMM_time = 0.0;
	Vector_time = 0.0;

	strcpy(Final_Vectors_FileName,"SubjectVectors.txt"); 
	z = fopen(Final_Vectors_FileName, "wt");
	if(!z){
		printf("Error %s\n",Final_Vectors_FileName);
		return -2;
	} 
	
	// Reading the sub-directory names from the input directory
	NbDirectories = GetFileNamesFromDirectory(MesArg[1],FullDirectoryNames,DirectoryNames);
	
	for(l=0;(l < NbDirectories);l++){
		for(m=0;(m < 4);m++){
			// Constructing the name of the current directory
			strcpy(Current_DirectoryName,FullDirectoryNames[l]); 
			strcat(Current_DirectoryName,Configurations[m]); 

			// Reading the file names from the current directory
			NbStatos = GetFileNamesFromDirectory(Current_DirectoryName,FullFileNames,FileNames);
			if(NbStatos > 0){
				// Creating the sub-directory where the result files will be saved
				strcpy(Results_DirectoryName,Current_DirectoryName); 
				strcat(Results_DirectoryName,"\\Results"); 
				k = mkdir(Results_DirectoryName);
				if(k != 0){
					if (errno != EEXIST) {
						perror("Error mkdir");
						return -1;
					}
				}
	
				// Creating the file where the vectors will be saved
				strcpy(MCs_FileName,Results_DirectoryName); 
				strcat(MCs_FileName,"\\MCs.txt"); 
				f = fopen(MCs_FileName, "wt");
				if(!f){
					printf("Error %s\n",MCs_FileName);
					return -2;
				} 

				// Creating the file where the separate vectors will be saved
				strcpy(Vectors_FileName,Results_DirectoryName); 
				strcat(Vectors_FileName,"\\AllVectors.txt"); 
				g = fopen(Vectors_FileName, "wt");
				if(!g){
					printf("Error %s\n",Vectors_FileName);
					return -2;
				} 	


				// Creating the file where the unique mean vector will be saved
				strcpy(Mean_Vector_FileName,Results_DirectoryName); 
				strcat(Mean_Vector_FileName,"\\MeanVector.txt"); 
				h = fopen(Mean_Vector_FileName, "wt");
				if(!h){
					printf("Error %s\n",Mean_Vector_FileName);
					return -2;
				} 	
				
				for(j=0;(j < NbSamples+3);j++){
					MeanVect[j] = 0.0;
				}
			}
			
			for(j=0;(j<NbStatos);j++){
				MaxAngle = 0.0;
				MaxDist = 0.0;

				overall_t = clock(); 
				Read_One_Statokinesigram_TimeShift(FullFileNames[j],&S,FreqDiv,TimeShift,&MaxAngle,&MaxDist,NbSamples,&f);
				overall_t = clock() - overall_t;
				Reading_time += ((double)overall_t)*1000.0/CLOCKS_PER_SEC;

				overall_t = clock(); 
				InitialHMM(S.TheMCs,&Lambda_Init,S.NbMCs,NbSamples);
				Baum_Welch_Multi(Lambda_Init,S.TheSequences,S.SeqLen,epsilon,MaxIterations,&Lambda,S.NbMCs,Threshold);
				overall_t = clock() - overall_t;
				HMM_time += ((double)overall_t)*1000.0/CLOCKS_PER_SEC;
				//printf("%i iterations\n",iterations);
				printf(".");

				overall_t = clock(); 
				VectorOfHMM(Lambda,Vect);
				Vect[Lambda.M] = MaxAngle;
				Vect[Lambda.M+1] = MaxDist;
				overall_t = clock() - overall_t;
				Vector_time += ((double)overall_t)*1000.0/CLOCKS_PER_SEC;

				for(i=0;(i < Lambda.M+2);i++){
					if(i != Lambda.M+1){
						fprintf(g,"%lf,",Vect[i]);
					}else{
						fprintf(g,"%lf\n",Vect[i]);
					}
				}
				SumOfVectors(MeanVect,Vect,Lambda.M+2);	
				
				if(SaveHMMs == 1){ 
					if(j == 0){
						// Creating the sub-directory where the HMMs will be saved
						strcpy(HMM_DirectoryName,Current_DirectoryName); 
						strcat(HMM_DirectoryName,"\\HMMs"); 
						k = mkdir(HMM_DirectoryName);
						if(k != 0){
							if (errno != EEXIST) {
								perror("Error mkdir");
								return -1;
							}
						}
					}
			
					// Saving the initial HMMs
					strcpy(HMM_FileName,HMM_DirectoryName); 
					strcat(HMM_FileName,"\\"); 
					strcat(HMM_FileName,"Lambda_Init_");
					strcat(HMM_FileName,FileNames[j]);
					k = strlen(HMM_FileName);
					HMM_FileName[k-4] = 0;
					strcat(HMM_FileName,".bin");
					SaveHMMtxt(Lambda_Init,HMM_FileName);
			
					// Saving the trained HMMs
					strcpy(HMM_FileName,HMM_DirectoryName); 
					strcat(HMM_FileName,"\\"); 
					strcat(HMM_FileName,"Lambda_");
					strcat(HMM_FileName,FileNames[j]);
					k = strlen(HMM_FileName);
					HMM_FileName[k-4] = 0;
					strcat(HMM_FileName,".dat");
					SaveHMMtxt(Lambda,HMM_FileName);
				}
			}

			for(j=0;(j < NbSamples+3);j++){
				if((m == 0)&&(j == 0)){
					fprintf(z,"[%s],",DirectoryNames[l]);
				}
				if(NbStatos > 0){
					// Saving the mean vector of the current configuration
					if(j != NbSamples+2){
						fprintf(h,"%lf,",MeanVect[j]/NbStatos);
					}else{
						fprintf(h,"%lf\n",MeanVect[j]/NbStatos);
					}
				
					// Saving the concatenation of the mean vectors of all the configurations
					if((j != NbSamples+2)||(m != 3)){
						fprintf(z,"%lf,",MeanVect[j]/NbStatos);
					}else{
						fprintf(z,"%lf\n",MeanVect[j]/NbStatos);
					}
				}else{
					// Saving the null vector in the concatenation of the mean vectors of all the configurations
					if((j != NbSamples+2)||(m != 3)){
						fprintf(z,"0,");
					}else{
						fprintf(z,"0\n");
					}
				}
			}
			fclose(f);
			fclose(g);
			fclose(h);
		}	
	}
	
	printf("\nData reading time = %lf ms\n",Reading_time);
	printf("HMM ititialization and training time = %lf ms\n",HMM_time);
	printf("Vector computation time = %lf ms\n",Vector_time);

	fclose(z);
	
	return 0;
}