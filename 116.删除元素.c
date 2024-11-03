#include<stdio.h>
void del(int a[],int n,int i);
void PrintArr(int a[],int n);
int main()
{
	int n,i;
	scanf("%d",&n);
	int A[n];
	for(i=0;i<n;i++)
	{
		scanf("%d",&A[i]);
	}
	int x;
	scanf("%d",&x);
	del(A,n,x);
	PrintArr(A,n-1);
	
	return 0;
}
void del(int a[],int n,int i){
	int b[n-1],j,h,g;
	for(j=0;j<i;j++){
		b[j]=a[j];
	}
	for(h=i+1;h<n;h++){
		b[j]=a[h];
		j++;
	}
	for(g=1;g<n-1;g++){
		a[g]=b[g];
	}
	
}
void PrintArr(int a[],int n){
	int i;
	for(i=0;i<n;i++){
		printf("%d ",a[i]);
	}
}
