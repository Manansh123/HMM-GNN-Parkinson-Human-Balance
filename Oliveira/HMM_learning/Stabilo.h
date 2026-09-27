#include "hmm.h"

#define MaxRecords 6000
#define MaxFilesInDirectory 2000
#define MaxVectors MaxFilesInDirectory

// An observation of a Markov chain
typedef struct Couple{      
	int State;                    	 
	int Symbol;                         
} Couple;

// A Markov chain
typedef struct MarkovChain{      
	int NbCouples;                
	Couple Elements[MaxLenMC];        
} MC;


// A Statokinesigram
typedef struct Statokinesigram{      
	int n;                    	 
	double COPx[MaxRecords];  
	double COPy[MaxRecords];  
	MC TheMCs[MaxMCs];
	int TheSequences[MaxMCs][MaxLenMC];
	int SeqLen[MaxMCs];
	int NbMCs;
	double MaxAngle;
	double MaxDist;
} Stato;

// Displays the content of a Markov chain
void DisplayMC(MC Delta){
	int i;
	
	for(i=0;(i<Delta.NbCouples);i++){
		printf("(%i,%i)",Delta.Elements[i].State,Delta.Elements[i].Symbol);
	}
	printf("\n");
	return;
}

// Prints the content of a Markov chain in a file
void DisplayMCfile(MC Delta, FILE ** f){
	int i;
	
	for(i=0;(i<Delta.NbCouples);i++){
		fprintf((*f),"(%i,%i)",Delta.Elements[i].State,Delta.Elements[i].Symbol);
	}
	fprintf((*f),"\n");
	return;
}


// Angle_XY
double Angle_XY(double X[2], double Y[2]){
	double Cosinus,A, Scal;
	double Norm_X, Norm_Y;
	
	Norm_X = sqrt(X[0]*X[0] + X[1]*X[1]);
	Norm_Y = sqrt(Y[0]*Y[0] + Y[1]*Y[1]);
	
	Scal = X[0]*Y[0] + X[1]*Y[1];
	Cosinus = (Scal/(Norm_X*Norm_Y));
	
	A = acos(Cosinus);
	
	if((Cosinus > 0)&&(1.0 - Cosinus < 1e-20)){
		Cosinus = 1.0;
	}
	if((Cosinus < 0)&&(Cosinus + 1.0 < 1e-20)){
		Cosinus = -1.0;
	}
	//printf("NormX=%lf,NormY=%lf,Scal=%lf,Cos=%lf,Acos=%lf\t",Norm_X,Norm_Y,Scal,Cosinus,A);
	return (A*180.0)/M_PI;
}

// Returns the value of the sample corresponding to a real number from [0,100]
int Sampling(double x, int NbSamples){
	int k;
	double Bsup, Binf;
	
	if(x == 0.0) return 0;
	for(k=1;(k <= NbSamples);k++){
		Binf = 100.0*(k-1);
		Bsup = 100.0*k;
		Binf = Binf/NbSamples;
		Bsup = Bsup/NbSamples;
		if((Binf < x)&&(x <= Bsup)){
			return k;
		}
	}
	return 0;
}


// Reads the Statokinesigram data with a constant time shift of 1
int Read_One_Statokinesigram(char * Statokinesigram_FileName, Stato * S, int step, double * MaxAngle, double * MaxDist, int NbSamples, FILE ** g){
	FILE * f;
	double x,y,t,GRFml,GRFap,GRFv,Mml,Map,Mv,Mfree,COPx,COPy, Angle, Dist, X[2],Y[2];
	int n,i,j,CurrentMC,LenCurrentMC;
	double Distances[MaxRecords], Angles[MaxRecords];
	char c;
	
	f = fopen(Statokinesigram_FileName, "rt");
	if(!f){
		printf("%s\n",Statokinesigram_FileName);
		return -1;
	} 
	fread(&c,sizeof(char),1,f);
	while(c != '\n'){
		fread(&c,sizeof(char),1,f);
	}
	fscanf(f,"%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n",&t,&GRFml,&GRFap,&GRFv,&Mml,&Map,&Mv,&Mfree,&COPx,&COPy);
	(*S).COPx[0] = COPx;
	(*S).COPy[0] = COPy;

	n = 1;
	i = 1;
	while(!feof(f)){
		if(!feof(f)){
			fscanf(f,"%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n",&t,&GRFml,&GRFap,&GRFv,&Mml,&Map,&Mv,&Mfree,&COPx,&COPy);
			i++;
			if(step == 1){
				(*S).COPx[n] = COPx;
				(*S).COPy[n] = COPy;
				n++;
			}else if(i % step == 1){
				(*S).COPx[n] = COPx;
				(*S).COPy[n] = COPy;
				n++;
			}
		}
	}
	(*S).n = n;
	fclose(f);
	
	for(j=1;(j < n);j++){
		X[0] = (*S).COPx[j-1];
		X[1] = (*S).COPy[j-1];
		Y[0] = (*S).COPx[j];
		Y[1] = (*S).COPy[j];
		
		Angle = Angle_XY(X,Y);
		if(Angle > (*MaxAngle)){
			(*MaxAngle) = Angle;
		}
		
		Dist = sqrt((X[0]-Y[0])*(X[0]-Y[0]) + (X[1]-Y[1])*(X[1]-Y[1]));
		if(Dist > (*MaxDist)){
			(*MaxDist) = Dist;
		}
		Angles[j-1] = Angle;
		Distances[j-1] = Dist;
	}

	CurrentMC = 0;
	LenCurrentMC = 0;
	for(j=0;(j < n-1);j++){
		if(LenCurrentMC == MaxLenMC){
			(*S).TheMCs[CurrentMC].NbCouples = LenCurrentMC;
			(*S).SeqLen[CurrentMC] = LenCurrentMC;
			(*S).TheMCs[CurrentMC+1].Elements[0].State = (*S).TheMCs[CurrentMC].Elements[LenCurrentMC-1].State;
			(*S).TheMCs[CurrentMC+1].Elements[0].Symbol = (*S).TheMCs[CurrentMC].Elements[LenCurrentMC-1].Symbol;
			(*S).TheSequences[CurrentMC+1][0] = (*S).TheMCs[CurrentMC].Elements[LenCurrentMC-1].Symbol;
			CurrentMC++;
			LenCurrentMC = 1;
		}
		x = 100.0*(Angles[j])/(*MaxAngle);
		(*S).TheMCs[CurrentMC].Elements[LenCurrentMC].State = Sampling(x,NbSamples);
		y = 100.0*(Distances[j])/(*MaxDist);
		(*S).TheMCs[CurrentMC].Elements[LenCurrentMC].Symbol = Sampling(y,NbSamples);
		(*S).TheSequences[CurrentMC][LenCurrentMC] = (*S).TheMCs[CurrentMC].Elements[LenCurrentMC].Symbol;
		LenCurrentMC++;
	}
	(*S).TheMCs[CurrentMC].NbCouples = LenCurrentMC;
	(*S).SeqLen[CurrentMC] = LenCurrentMC;
	(*S).NbMCs = CurrentMC+1;
	
	//printf("\n\n%i total samples\n",i);
	//printf("%i saved samples\n",n);
	//printf("MaxAngle = %lf\n",(*MaxAngle));
	//printf("MaxDist = %lf\n",(*MaxDist));
	
	fprintf(*g,"[%s]\n",Statokinesigram_FileName);
	fprintf(*g,"%i samples\n",n);
	fprintf(*g,"%i Markov chains\n\n",CurrentMC+1);
	for(j=0;(j<=CurrentMC);j++){
		DisplayMCfile((*S).TheMCs[j],g);
	}
	fprintf(*g,"__________________________________________________________________________________________\n");
	return CurrentMC;
}


// Reads the Statokinesigram data with a variable time shift
int Read_One_Statokinesigram_TimeShift(char * Statokinesigram_FileName, Stato * S, int step, int shift, double * MaxAngle, double * MaxDist, int NbSamples, FILE ** g){
	FILE * f;
	double x,y,t,GRFml,GRFap,GRFv,Mml,Map,Mv,Mfree,COPx,COPy, Angle, Dist, X[2],Y[2];
	int n,i,j,CurrentMC,LenCurrentMC;
	double Distances[MaxRecords], Angles[MaxRecords];
	char c;
	
	f = fopen(Statokinesigram_FileName, "rt");
	if(!f){
		printf("%s\n",Statokinesigram_FileName);
		return -1;
	} 
	// Skipping the first line
	fread(&c,sizeof(char),1,f);
	while(c != '\n'){
		fread(&c,sizeof(char),1,f);
	}
	// Reading the first data
	fscanf(f,"%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n",&t,&GRFml,&GRFap,&GRFv,&Mml,&Map,&Mv,&Mfree,&COPx,&COPy);
	(*S).COPx[0] = COPx;
	(*S).COPy[0] = COPy;

	n = 1;
	i = 1;
	while(!feof(f)){
		if(!feof(f)){
			fscanf(f,"%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n",&t,&GRFml,&GRFap,&GRFv,&Mml,&Map,&Mv,&Mfree,&COPx,&COPy);
			i++;
			if(step == 1){
				(*S).COPx[n] = COPx;
				(*S).COPy[n] = COPy;
				n++;
			}else if(i % step == 1){
				(*S).COPx[n] = COPx;
				(*S).COPy[n] = COPy;
				n++;
			}
		}
	}
	(*S).n = n;
	fclose(f);
	
	for(j=shift;(j < n);j++){
		X[0] = (*S).COPx[j-shift];
		X[1] = (*S).COPy[j-shift];
		Y[0] = (*S).COPx[j];
		Y[1] = (*S).COPy[j];
		
		Angle = Angle_XY(X,Y);
		if(Angle > (*MaxAngle)){
			(*MaxAngle) = Angle;
		}
		
		Dist = sqrt((X[0]-Y[0])*(X[0]-Y[0]) + (X[1]-Y[1])*(X[1]-Y[1]));
		if(Dist > (*MaxDist)){
			(*MaxDist) = Dist;
		}
		Angles[j-shift] = Angle;
		Distances[j-shift] = Dist;
	}

	CurrentMC = 0;
	LenCurrentMC = 0;
	for(j=0;(j < n-shift);j++){
		if(LenCurrentMC == MaxLenMC){
			(*S).TheMCs[CurrentMC].NbCouples = LenCurrentMC;
			(*S).SeqLen[CurrentMC] = LenCurrentMC;
			(*S).TheMCs[CurrentMC+1].Elements[0].State = (*S).TheMCs[CurrentMC].Elements[LenCurrentMC-1].State;
			(*S).TheMCs[CurrentMC+1].Elements[0].Symbol = (*S).TheMCs[CurrentMC].Elements[LenCurrentMC-1].Symbol;
			(*S).TheSequences[CurrentMC+1][0] = (*S).TheMCs[CurrentMC].Elements[LenCurrentMC-1].Symbol;
			CurrentMC++;
			LenCurrentMC = 1;
		}
		x = 100.0*(Angles[j])/(*MaxAngle);
		(*S).TheMCs[CurrentMC].Elements[LenCurrentMC].State = Sampling(x,NbSamples);
		y = 100.0*(Distances[j])/(*MaxDist);
		(*S).TheMCs[CurrentMC].Elements[LenCurrentMC].Symbol = Sampling(y,NbSamples);
		(*S).TheSequences[CurrentMC][LenCurrentMC] = (*S).TheMCs[CurrentMC].Elements[LenCurrentMC].Symbol;
		LenCurrentMC++;
	}
	(*S).TheMCs[CurrentMC].NbCouples = LenCurrentMC;
	(*S).SeqLen[CurrentMC] = LenCurrentMC;
	(*S).NbMCs = CurrentMC+1;
	
	//printf("\n\n%i total samples\n",i);
	//printf("%i saved samples\n",n);
	//printf("MaxAngle = %lf\n",(*MaxAngle));
	//printf("MaxDist = %lf\n",(*MaxDist));
	
	fprintf(*g,"[%s]\n",Statokinesigram_FileName);
	fprintf(*g,"%i samples\n",n);
	fprintf(*g,"%i Markov chains\n\n",CurrentMC+1);
	for(j=0;(j<=CurrentMC);j++){
		DisplayMCfile((*S).TheMCs[j],g);
	}
	fprintf(*g,"__________________________________________________________________________________________\n");
	return CurrentMC;
}


// Collects the names of the elements (file, sub-directory) of the directory 'DirectoryName'
int GetFileNamesFromDirectory(char DirectoryName[50], char FullFileNames[MaxFilesInDirectory][100], char FileNames[MaxFilesInDirectory][50]) 
{ 
    struct dirent *de; 
	int i,k;
	char F_Name[100];
  
    DIR *dr = opendir(DirectoryName); 
    if(!dr) return -1; 
	
	k = 0;
    for(i=0;((de = readdir(dr)) != NULL);i++){ 
        strcpy(F_Name,DirectoryName); 
        strcat(F_Name,"\\"); 
		if(de->d_name[0] != '.'){
			strcat(F_Name,de->d_name); 
			strcpy(FullFileNames[k],F_Name);
			strcpy(FileNames[k],de->d_name);
			k++;
		}
	}
    closedir(dr);     
    return k; 
}


// Initializes the HMM corresponding to a Statokinesigram
void InitialHMM(MC TheMCs[MaxMCs], HMM * Lambda, int NbMCs, int NbSamples){
	int i,j,k,State,NextState,Symbol,From[MaxStates],UseOfState[MaxStates];
	double Sum;
	
	(*Lambda).N = NbSamples + 1;                 
	(*Lambda).M = NbSamples + 1;                 
	
	for(i=0;(i < (*Lambda).N);i++){
		(*Lambda).Pi[i] = 0.0;
		From[i] = 0;
		UseOfState[i] = 0;
		for(j=0;(j < (*Lambda).N);j++){
			(*Lambda).A[i][j] = 0.0;
		}
		for(j=0;(j < (*Lambda).M);j++){
			(*Lambda).B[i][j] = 0.0;
		}
	}

	for(i=0;(i < NbMCs);i++){
		for(j=0;(j < TheMCs[i].NbCouples);j++){
			if(j != TheMCs[i].NbCouples -1){
				State = TheMCs[i].Elements[j].State;
				NextState = TheMCs[i].Elements[j+1].State;
				Symbol = TheMCs[i].Elements[j].Symbol;

				From[State] += 1;
				UseOfState[State] += 1;
				(*Lambda).A[State][NextState] += 1.0;
				(*Lambda).B[State][Symbol] += 1.0;
					
				if(j == 0){
					(*Lambda).Pi[State] += 1.0;
				}
			}else{
				State = TheMCs[i].Elements[j].State;
				Symbol = TheMCs[i].Elements[j].Symbol;

				UseOfState[State] += 1;
				(*Lambda).B[State][Symbol] += 1.0;
			}
				
		}	
	}	
	
	for(i=0;(i < (*Lambda).N);i++){
		(*Lambda).Pi[i] /= (NbMCs+Epsi);
		for(j=0;(j < (*Lambda).N);j++){
			(*Lambda).A[i][j] /= (From[i]+Epsi);
		}
	}
	for(j=0;(j < (*Lambda).N);j++){
		for(k=0;(k < (*Lambda).M);k++){
			(*Lambda).B[j][k] /= (UseOfState[j]+Epsi);
		}
	}

	Sum = 0.0;
	for(i=0;(i < (*Lambda).N);i++){
		Sum += (*Lambda).Pi[i];
	}	
	Sum = fabs(1.0 - Sum)/(*Lambda).N;
	for(i=0;(i < (*Lambda).N);i++){
		(*Lambda).Pi[i] += Sum;
	}	
	for(i=0;(i < (*Lambda).N);i++){
		Sum = 0.0;
		for(j=0;(j < (*Lambda).N);j++){
			Sum += (*Lambda).A[i][j];
		}
		Sum = fabs(1.0 - Sum)/(*Lambda).N;
		for(j=0;(j < (*Lambda).N);j++){
			(*Lambda).A[i][j] += Sum;
		}	
	}	
		
	for(j=0;(j < (*Lambda).N);j++){
		Sum = 0.0;
		for(k=0;(k < (*Lambda).M);k++){
			Sum += (*Lambda).B[j][k];
		}
		Sum = fabs(1.0 - Sum)/(*Lambda).M;
		for(k=0;(k < (*Lambda).M);k++){
			(*Lambda).B[j][k] += Sum;
		}	
	}
		
	StationaryDistribution(Lambda);
	
	
	return;
}	

// Sum of two vectors
void SumOfVectors(double X[MaxSymbols+2], double Y[MaxSymbols+2], int n){
	int k;
	
	for(k=0;(k < n);k++){
		X[k] += Y[k];
	}	
	return;
}
