#include<stdio.h>
int main()
{
	int a[1001],i,n,b=0,j,t;
	scanf("%d",&n);
	for(i=0;i<n;i++){
		scanf("%d",&a[i]);
	}
	for(i=0;i<n;i++){
		if(a[i]!=-1) b++;
		for(j=i+1;j<n;j++){
			if(a[i]==a[j]) 
				a[j]=-1;
		}
	}
	for(i=0;i<n;i++)
	for(j=i+1;j<n;j++){
		if(a[i]>a[j]){
			t=a[i];
			a[i]=a[j];
			a[j]=t;
		}
	}
	
	printf("%d\n",b);
	for(i=0;i<n;i++){
		if(a[i]!=-1) printf("%d ",a[i]);
	}
	
	return 0;
}
