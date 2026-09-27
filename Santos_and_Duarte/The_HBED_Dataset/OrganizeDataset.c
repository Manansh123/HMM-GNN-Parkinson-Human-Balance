#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>

/*
____________________________________________________________________________________________   
   Compilation command line:

   gcc -Wall -Wl,--stack,536870912 -o OrganizeDataset.exe OrganizeDataset.c
________________________________________________________________________________________________   
   Execution command line:

   OrganizeDataset.exe FilesList OrganizedDirectory OriginalDirectory
________________________________________________________________________________________________   
   Example: 
   
   OrganizeDataset.exe FilesList.csv Organized Original
________________________________________________________________________________________________   
*/


int main(int NbArg, char ** MesArg) {
	FILE *f, *g, *h;
	int i,j,Id;
	char Vision[10],Surface[10], Subject_Id[10],File_Name[100], Current_DirectoryName[100];
	char c,x,Ch[100],Full_File_Name[100],Current_File_Name[100];
	char Organized[100], Original[100], FilesList[100];
	
	strcpy(FilesList,MesArg[1]);
	strcpy(Organized,MesArg[2]);
	strcpy(Original,MesArg[3]);
	
	// Creating the Organized directory
	mkdir(Organized);
	
	// Opening the FileList
	x = 'a';
	f = fopen(FilesList, "rt");
	if(!f){
		printf("Error %s\n",FilesList);
		return -1;
	} 

	// Reading the file name
	j = 0;
	do{
		fread(&c,sizeof(char),1,f);
		if(c != ';'){
			File_Name[j] = c;
			j++;
		}
	}while(c != ';');
	File_Name[j] = 0;

	// Reading the subject Id
	fscanf(f,"%i;",&i);
	
	// Reading the Surface
	j = 0;
	do{
		fread(&c,sizeof(char),1,f);
		if(c != ';'){
			Surface[j] = c;
			j++;
		}
	}while(c != ';');
	Surface[j] = 0;
	
	// Reading the Vision
	j = 0;
	do{
		fread(&c,sizeof(char),1,f);
		if(c != '\n'){
			Vision[j] = c;
			j++;
		}
	}while(c != '\n');
	Vision[j] = 0;
	
	printf("%s,%i,%s,%s\n",File_Name,i,Surface,Vision);

	// Creating the subject directory
	itoa(i,Subject_Id,10);
	strcpy(Current_DirectoryName,Organized); 
	strcat(Current_DirectoryName,"\\"); 
	strcat(Current_DirectoryName,Subject_Id); 
	mkdir(Current_DirectoryName);
	
	// Constructing the name of the current directory
	strcpy(Ch,Current_DirectoryName); 
	strcat(Ch,"\\Firm"); 
	mkdir(Ch);
		
	strcpy(Ch,Current_DirectoryName); 
	strcat(Ch,"\\Foam"); 
	mkdir(Ch);
		
	strcpy(Ch,Current_DirectoryName); 
	strcat(Ch,"\\Firm\\Open"); 
	mkdir(Ch);
		
	strcpy(Ch,Current_DirectoryName); 
	strcat(Ch,"\\Firm\\Closed"); 
	mkdir(Ch);
		
	strcpy(Ch,Current_DirectoryName); 
	strcat(Ch,"\\Foam\\Open"); 
	mkdir(Ch);
	
	strcpy(Ch,Current_DirectoryName); 
	strcat(Ch,"\\Foam\\Closed"); 
	mkdir(Ch);

	Id = i;
	while(!feof(f)){
		if(!feof(f)){
			if(i == Id){
				strcpy(Full_File_Name,Organized); 
				strcat(Full_File_Name,"\\"); 
				strcat(Full_File_Name,Subject_Id); 
				strcat(Full_File_Name,"\\"); 
				strcat(Full_File_Name,Surface); 
				strcat(Full_File_Name,"\\"); 
				strcat(Full_File_Name,Vision); 
				strcat(Full_File_Name,"\\"); 
				strcat(Full_File_Name,File_Name); 
				strcat(Full_File_Name,".txt"); 
				
				g = fopen(Full_File_Name, "wt");
				if(!g){
					printf("Error + %s\n",Full_File_Name);
					return -2;
				}	 

				strcpy(Current_File_Name,Original); 
				strcat(Current_File_Name,"\\"); 
				strcat(Current_File_Name,File_Name); 
				strcat(Current_File_Name,".txt"); 

				h = fopen(Current_File_Name, "rt");
				if(!h){
					printf("Error - %s\n",Current_File_Name);
					return -2;
				} 	
				
				// Copy of the file data
				while(!feof(h)){
					if(!feof(h)){
						fread(&c,sizeof(char),1,h);
						if((x != '\n')||(c != '\n')){
							fwrite(&c,sizeof(char),1,g);
						}
						x = c;
					}
				}	
				fclose(g);
				fclose(h);
				

				// Reading the file name
				j = 0;
				do{
					fread(&c,sizeof(char),1,f);
					if(c != ';'){
						File_Name[j] = c;
						j++;
					}
				}while(c != ';');
				File_Name[j] = 0;

				// Reading the subject Id
				fscanf(f,"%i;",&i);
	
				// Reading the Surface
				j = 0;
				do{
					fread(&c,sizeof(char),1,f);
					if(c != ';'){
						Surface[j] = c;
						j++;
					}
				}while(c != ';');
				Surface[j] = 0;
	
				// Reading the Vision
				j = 0;
				do{
					fread(&c,sizeof(char),1,f);
					if(c != '\n'){
						Vision[j] = c;
						j++;
					}
				}while(c != '\n');
				Vision[j] = 0;
				printf("%s,%i,%s,%s\n",File_Name,i,Surface,Vision);
				
			}else{
				// Creating the subject directory
				itoa(i,Subject_Id,10);
				strcpy(Current_DirectoryName,Organized); 
				strcat(Current_DirectoryName,"\\");
				strcat(Current_DirectoryName,Subject_Id); 
				mkdir(Current_DirectoryName);
	
				// Constructing the name of the current directory
				strcpy(Ch,Current_DirectoryName); 
				strcat(Ch,"\\Firm"); 
				mkdir(Ch);
		
				strcpy(Ch,Current_DirectoryName); 
				strcat(Ch,"\\Foam"); 
				mkdir(Ch);
		
				strcpy(Ch,Current_DirectoryName); 
				strcat(Ch,"\\Firm\\Open"); 
				mkdir(Ch);
		
				strcpy(Ch,Current_DirectoryName); 
				strcat(Ch,"\\Firm\\Closed"); 
				mkdir(Ch);
		
				strcpy(Ch,Current_DirectoryName); 
				strcat(Ch,"\\Foam\\Open"); 
				mkdir(Ch);
	
				strcpy(Ch,Current_DirectoryName); 
				strcat(Ch,"\\Foam\\Closed"); 
				mkdir(Ch);

				Id = i;
			}	
		}
	}
	
	fclose(f);
	return 0;
}
