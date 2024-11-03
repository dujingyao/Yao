#include<stdio.h>
int MinIndex(int a[], int n); 
int MaxIndex(int a[], int n);
void PrintArr(int a[],int n);
int main()
{
	int n;
	scanf("%d",&n);
	int a[n],i;
	for(i=0;i<n;i++){
		scanf("%d",&a[i]);
	}
	int min,max,j;
	min=MinIndex(a,n);
	j=a[0];
	a[0]=a[min];
	a[min]=j;
	max=MaxIndex(a,n);
	j=a[max];
	a[max]=a[n-1];
	a[n-1]=j;
	PrintArr(a,n);
	
	return 0;
}
void PrintArr(int a[],int n){
	int i;
	for(i=0;i<n;i++){
		printf("%d ",a[i]);
	}
}
int MinIndex(int a[],int n){
	int i,x;
	int min=a[0];
	for(i=0;i<n;i++){
		if(a[i]<min||a[i]==min) min=a[i],x=i;
	}
	return x;
}
int MaxIndex(int a[],int n){
	int i,x;
	int max=a[0];
	for(i=0;i<n;i++){
		if(a[i]>max||a[i]==max) max=a[i],x=i;
	}
	return x;
}
