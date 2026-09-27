#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/*
____________________________________________________________________________________________   
   Compilation command line:

   gcc -Wall -o MakeArffs.exe MakeArffs.c
________________________________________________________________________________________________   
   Execution command line:

   MakeArffs.exe InfoFile VectorFile NbComponents
________________________________________________________________________________________________   
   Example: 
   
   MakeArffs.exe BDSinfoSmallClean.csv Vectors.txt 52
________________________________________________________________________________________________   
*/

int main(int NbArg, char ** MesArg) {
	FILE *f, *g, *Fage,*Fbest;
	int i,Id,Best_T,NbComponents;
	char c,AgeGroup;
	
	NbComponents = atoi(MesArg[3]);
	
	// Opening the Info file
	f = fopen(MesArg[1], "rt");
	if(!f){
		printf("Error %s\n",MesArg[1]);
		return -1;
	} 
	// Skipping the first line
	do{
		fread(&c,sizeof(char),1,f);
	}while(c != '\n');
	
	// Opening the file containing the vectors
	g = fopen(MesArg[2], "rt");
	if(!g){
		printf("Error %s\n",MesArg[2]);
		return -1;
	} 
	
	// Creating the relations and the classes
	Fage = fopen("Age.arff", "wt");
	if(!Fage){
		printf("Error Age.arff\n");
		return -1;
	} 
	fprintf(Fage,"@RELATION AgeGroup\n");
	fprintf(Fage,"@ATTRIBUTE class {young,old}\n");
	
	Fbest = fopen("BEST.arff", "wt");
	if(!Fbest){
		printf("Error BEST.arff\n");
		return -1;
	} 
	fprintf(Fbest,"@RELATION BEST\n");
	fprintf(Fbest,"@ATTRIBUTE class {low_fall_risk,high_fall_risk}\n");
	
	// Writing the other attributes
	for(i = 1; i <= NbComponents; i++){
		fprintf(Fage,"@ATTRIBUTE x%i NUMERIC\n",i);
		fprintf(Fbest,"@ATTRIBUTE x%i NUMERIC\n",i);
	}
	
	// Data start here
	fprintf(Fage,"@DATA\n");
	fprintf(Fbest,"@DATA\n");
	
	do{
		if(!feof(f)){
			// Reading the next Info
			fscanf(f,"%i;%c;%i\n",&Id,&AgeGroup,&Best_T);
		
			// AgeGroup
			if(AgeGroup == 'y'){
				fprintf(Fage,"young,");
			}else{
				fprintf(Fage,"old,");
			}
		
			// BEST
			if(Best_T <= 20){
				fprintf(Fbest,"high_fall_risk,");
			}else{
				fprintf(Fbest,"low_fall_risk,");
			}
			
			//Copying the vector components
			fscanf(g,"%i,",&i);
			if(i == Id){
				do{
					fread(&c,sizeof(char),1,g);
					fwrite(&c,sizeof(char),1,Fage);
					fwrite(&c,sizeof(char),1,Fbest);
				}while(c != '\n');
			}else{
				printf("Wrong subjects (%i,%i)\n",i,Id);
				return -1;
			}
		}
	}while(!feof(f));
	
	// Closing all the files
	fclose(f);
	fclose(g);
	fclose(Fage);
	fclose(Fbest);
	
	return 0;
}

