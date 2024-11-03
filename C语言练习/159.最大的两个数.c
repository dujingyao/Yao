#include<stdio.h>
void LargestTwo(int a[],int n,int *pfirst,int *psecond);
int main()
{
	int n,i,a[1001];
	scanf("%d",&n);
	for(i=0;i<n;i++){
		scanf("%d",&a[i]);
	}
	int *pfirst=&a[0],*psecond=&a[1];
	LargestTwo(a,n,pfirst,psecond);
	printf("%d %d",*pfirst,*psecond);
	
	return 0;
}
void LargestTwo(int a[],int n,int *pfirst,int *psecond){
	int i,j,t;
	for(i=0;i<n;i++){
		for(j=i+1;j<n;j++){
			if(a[i]<a[j]){
				t=a[i];
				a[i]=a[j];
				a[j]=t;
			}
		}
	}
	pfirst=&a[0];
	psecond=&a[1];
}
