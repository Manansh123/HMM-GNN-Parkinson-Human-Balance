#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/*
____________________________________________________________________________________________   
   Compilation command line:

   gcc -Wall -o MergeArffs.exe MergeArffs.c
________________________________________________________________________________________________   
   Execution command line:

   MergeArffs.exe File1 File2 ResultFile
________________________________________________________________________________________________   
   Example:  
   
   MergeArffs.exe Age10_shift_1.arff Age10_shift_5.arff Age10_shift_1_5.arff
________________________________________________________________________________________________   
*/

int main(int NbArg, char ** MesArg) {
	FILE *f, *g, *h;
	int i,Nf, Ng, Nvec;
	char c,x,Class[100];
	
	// Opening the File1
	f = fopen(MesArg[1], "rt");
	if(!f){
		printf("Error %s\n",MesArg[1]);
		return -1;
	} 
	// Skipping the two first line
	do{
		fread(&c,sizeof(char),1,f);
	}while(c != '\n');
	do{
		fread(&c,sizeof(char),1,f);
	}while(c != '\n');
	
	// Opening the File2
	g = fopen(MesArg[2], "rt");
	if(!g){
		printf("Error %s\n",MesArg[2]);
		return -1;
	} 
	
	
	// Opening the ResultFile
	h = fopen(MesArg[3], "wt");
	if(!h){
		printf("Error %s\n",MesArg[3]);
		return -1;
	} 
	
	// Copy of the relation name from the File2 in the ResultFile
	do{
		fread(&c,sizeof(char),1,g);
		fwrite(&c,sizeof(char),1,h);
	}while(c != '\n');
	// Copy of the class names from the File2 in the ResultFile
	do{
		fread(&c,sizeof(char),1,g);
		fwrite(&c,sizeof(char),1,h);
	}while(c != '\n');
	
	// Counting the numeric attributes of the File1
	Nf = 0;
	do{
		fread(&c,sizeof(char),1,f);
		fread(&x,sizeof(char),1,f);
		if(x == 'A'){
			Nf++;
		}
		do{
			fread(&c,sizeof(char),1,f);
		}while(c != '\n');
	}while(x != 'D');
	
	// Counting the numeric attributes of the File2
	Ng = 0;
	do{
		fread(&c,sizeof(char),1,g);
		fread(&x,sizeof(char),1,g);
		if(x == 'A'){
			Ng++;
		}
		do{
			fread(&c,sizeof(char),1,g);
		}while(c != '\n');
	}while(x != 'D');
	
	// Writing the (Nf + Ng) numeric attributes names in the ResultFile
	for(i = 1; i <= (Nf + Ng); i++){
		fprintf(h,"@ATTRIBUTE x%i NUMERIC\n",i);
	}
	// Data start here in the ResultFile
	fprintf(h,"@DATA\n");
	Nvec = 0;
	do{
		fread(&c,sizeof(char),1,f);
		fread(&c,sizeof(char),1,g);
		
		if((!feof(f))&&(!feof(g))){
			// Copy of the current vector class name in ResultFile
			Class[0] = c;
			i = 1;
			do{
				fread(&c,sizeof(char),1,f);
				fread(&c,sizeof(char),1,g);
				if(c != ','){
					Class[i] = c;
					i++;
				}
			}while(c != ',');
			Class[i] = 0;
			fprintf(h,"%s,",Class);
			
			// Copy of the current vector attributes from the File1 in the ResultFile
			do{
				fread(&c,sizeof(char),1,f);
				if(c != '\n'){
					fwrite(&c,sizeof(char),1,h);
				}else{
					fprintf(h,",");
				}
			}while(c != '\n');
			
			// Copy of the current vector attributes from the File2 in the ResultFile
			do{
				fread(&c,sizeof(char),1,g);
				fwrite(&c,sizeof(char),1,h);
			}while(c != '\n');
			Nvec++;
		}
	}while((!feof(f))&&(!feof(g)));
	
	// Closing all the files
	fclose(f);
	fclose(g);
	fclose(h);
	
	printf("SUCCESS! copy of %i vectors\n",Nvec);
	
	return 0;
}

