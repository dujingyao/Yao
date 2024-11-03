#include<stdio.h>
int a[1000000]={0},b[1000000];
int main()
{
	int n,m,i,j;
	scanf("%d",&n);
	for(i=n-1;i>=0;i--){
		scanf("%d",&a[i]);
	}
	scanf("%d",&m);
	for(j=0;j<m;j++){
		scanf("%d",&b[j]);
	}
	int c[m+n],k=0;
	i=0,j=0;
	while(i<n&&j<m){
		if(a[i]>=b[j]){
			c[k++]=a[i++];
		}
		else{c[k++]=b[j++];
		}
	}
	while(i<n){
		c[k++]=a[i++];
	}
	while(j<m){
		c[k++]=b[j++];
	}
	for(k=0;k<m+n-1;k++)
	printf("%d ",c[k]);
	printf("%d",c[m+n-1]);
	
	return 0;
}

