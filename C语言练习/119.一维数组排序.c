#include<stdio.h>
void sort(int a[], int n);
void PrintArr(int a[],int n);
int main()
{
	int n,i;
	scanf("%d",&n);
	int a[n];
	for(i=0;i<n;i++){
		scanf("%d",&a[i]);
	}
	sort(a,n);
	PrintArr(a,n);
	
	return 0;
}
void PrintArr(int a[],int n){
	int i;
	for(i=0;i<n;i++){
		printf("%d ",a[i]);
	}
}
void sort(int a[],int n){
	int i,c,x=n;
	while(x>0){
		x--;
		for(i=0;i<n-1;i++){
			if(a[i]>a[i+1]){
				c=a[i];
				a[i]=a[i+1];
				a[i+1]=c;
			}
		}
	}
}
