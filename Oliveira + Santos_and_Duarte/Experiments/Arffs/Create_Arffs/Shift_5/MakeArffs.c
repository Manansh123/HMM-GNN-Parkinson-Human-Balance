#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define MaxComponents 100
#define MaxVectors 200

/*
____________________________________________________________________________________________   
   Compilation command line:

   gcc -Wall -o MakeArffs.exe MakeArffs.c
________________________________________________________________________________________________   
   Execution command line:

   MakeArffs.exe PositiveFile NegativeFile NoIllnessFile zcore
________________________________________________________________________________________________   
   Example: 
   
   MakeArffs.exe Positive_Vectors.txt Negative_Vectors.txt No_illness.txt 0
   MakeArffs.exe Positive_Vectors.txt Negative_Vectors.txt No_illness.txt 1
________________________________________________________________________________________________   
*/

// Reading the vector data
int ReadVectors(char * Positive_FileName, char * Negative_FileName, char * NoIllness_FileName, double Pos[MaxVectors][MaxComponents], double Neg[MaxVectors][MaxComponents], int * NbComponentsPos, int * NbComponentsNeg, int * NbVectorsPos, int * NbVectorsNeg){
	FILE *f;
	int i,j,k, NbVectors,n;
	char c, x[50];
	double NegTmp[MaxVectors][MaxComponents];
	
	
	// Opening the Positive file
	f = fopen(Positive_FileName, "rt");
	if(!f){
		printf("Error %s\n",Positive_FileName);
		return -1;
	} 

	// Reading the vectors in the Positive file
	NbVectors = 0;
	fscanf(f,"[%i]",&i);
	while(!feof(f)){
		n = 0;
		do{
			fread(&c,sizeof(char),1,f);
			if(c != '\n'){
				i = 0;
				do{
					fread(&c,sizeof(char),1,f);
					if((c != '\n')&&(c != ',')){
						x[i++] = c;
					}
				}while((c != '\n')&&(c != ','));
				if(i > 0){
					x[i] = 0;
					Pos[NbVectors][n++] = atof(x);
				}
			}
		}while(c != '\n');
		NbVectors++;
		fscanf(f,"[%i]",&i);
	}	
	// Closing the Positive file
	fclose(f);
	(*NbVectorsPos) = NbVectors;
	(*NbComponentsPos) = n;
	
	
	// Opening the Negative file
	f = fopen(Negative_FileName, "rt");
	if(!f){
		printf("Error %s\n",Negative_FileName);
		return -1;
	} 

	// Reading the vectors in the Negative file
	NbVectors = 0;
	fscanf(f,"%i",&i);
	while(!feof(f)){
		n = 0;
		do{
			fread(&c,sizeof(char),1,f);
			if(c != '\n'){
				i = 0;
				do{
					fread(&c,sizeof(char),1,f);
					if((c != '\n')&&(c != ',')){
						x[i++] = c;
					}
				}while((c != '\n')&&(c != ','));
				if(i > 0){
					x[i] = 0;
					NegTmp[NbVectors][n++] = atof(x);
				}
			}
		}while(c != '\n');
		NbVectors++;
		fscanf(f,"%i",&i);
	}	
	// Closing the Negative file
	fclose(f);
	
	// Opening the No_illness file
	f = fopen(NoIllness_FileName, "rt");
	if(!f){
		printf("Error %s\n",NoIllness_FileName);
		return -1;
	} 

	// Reading the indices of the subjects that have declared no illness
	fscanf(f,"%i\n",&i);
	k = 0;
	do{
		for(j=0;(j<n);j++){
			Neg[k][j] = NegTmp[i-1][j];
		}
		k++;
		fscanf(f,"%i\n",&i);
	}while(!feof(f));

	for(j=0;(j<n);j++){
		Neg[k][j] = NegTmp[i-1][j];
	}
	k++;
	
	// Closing the No_illness file
	fclose(f);
	(*NbVectorsNeg) = k;
	(*NbComponentsNeg) = n;
	
	return 0;
}

// z-score scalling for eventual batch effects correction
void zscore_scalling(double Vector[MaxVectors][MaxComponents], int NbVectors, int NbComponents) {
	int i,j;
	double mean, var, std;
	
    for (j = 0; j < NbComponents; j++) {
        mean = 0.0;
		var = 0.0;

        // Computing the mean
        for (i = 0; i < NbVectors; i++) {
            mean += Vector[i][j];
        }
        mean /= NbVectors;

        // Computing the variance
        for (i = 0; i < NbVectors; i++) {
            var += (Vector[i][j] - mean) * (Vector[i][j] - mean);
        }
        var /= NbVectors;
        std = (var > 0) ? sqrt(var) : 1.0;

        // Normalizing
        for (i = 0; i < NbVectors; i++) {
            Vector[i][j] = (Vector[i][j] - mean) / std;
        }
    }
}


int main(int NbArg, char ** MesArg) {
	FILE *h;
	int i,j,NbAttributes, NbComponentsPos, NbComponentsNeg;
	int zscore, NbVectorsPos, NbVectorsNeg;
	double Pos[MaxVectors][MaxComponents], Neg[MaxVectors][MaxComponents];
	
	// Reading the vector data
	ReadVectors(MesArg[1],MesArg[2],MesArg[3],Pos,Neg,&NbComponentsPos,&NbComponentsNeg,&NbVectorsPos,&NbVectorsNeg);	
	if(NbComponentsPos != NbComponentsNeg){
		return -1;
	}
	NbAttributes = NbComponentsPos;
	
	/*
		Creating the arff file for the Parkinson desease 
		with eventual application of z-score scalling
	*/	
	zscore = atoi(MesArg[4]);
	if(zscore != 0){
		zscore_scalling(Pos,NbVectorsPos,NbAttributes);
		zscore_scalling(Neg,NbVectorsNeg,NbAttributes);
		h = fopen("Parkinson_zscore.arff", "wt");
		if(!h){
			printf("Error Parkinson_zscore.arff\n");
			return -1;
		} 
	}else{
		h = fopen("Parkinson.arff", "wt");
		if(!h){
			printf("Error Parkinson.arff\n");
			return -1;
		} 
	}

	
	// Inserting the name of the relation
	fprintf(h,"@RELATION Parkinson\n");
	// Inserting the the  names of the classes
	fprintf(h,"@ATTRIBUTE class {Positive,Negative}\n");
	// Inserting the names of the attributes 
	for(i = 1; i <= NbAttributes; i++){
		fprintf(h,"@ATTRIBUTE x%i NUMERIC\n",i);
	}
	// Data start here
	fprintf(h,"@DATA\n");
	
	// Inserting the Positive vectors
	for(i=0;(i<NbVectorsPos);i++){
		fprintf(h,"Positive,");
		for(j = 0; j < NbAttributes; j++){
			if(j != NbAttributes - 1){
				fprintf(h,"%lg,",Pos[i][j]);
			}else{
				fprintf(h,"%lg\n",Pos[i][j]);
			}
		}
	}
	
	// Inserting the Negative vectors 
	for(i=0;(i<NbVectorsNeg);i++){
		fprintf(h,"Negative,");
		for(j = 0; j < NbAttributes; j++){
			if(j != NbAttributes - 1){
				fprintf(h,"%lg,",Neg[i][j]);
			}else{
				fprintf(h,"%lg\n",Neg[i][j]);
			}
		}
	}
	
	// Closing the arff file for the Parkinson desease
	fclose(h);

	return 0;
}

