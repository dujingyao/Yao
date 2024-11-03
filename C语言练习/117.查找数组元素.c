#include<stdio.h>
int find(int a[], int n, int x);
void del(int a[],int n,int i);
void PrintArr(int a[],int n);
int main()
{
	int n,i,m;
	scanf("%d",&n);
	int a[n];
	for(i=0;i<n;i++){
		scanf("%d",&a[i]);
	}
	scanf("%d",&m);
	int x;
	x=find(a,n,m);
	if(x==-1){
		printf("Not Found");
	}else {
		del(a,n,x);
		PrintArr(a,n-1);
	}
	
	
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
		printf("%4d",a[i]);
	}
}
int find(int a[],int n,int x){
	int i;
	for(i=0;i<n;i++){
		if(a[i]==x) return i;
	}
	return -1;
}
