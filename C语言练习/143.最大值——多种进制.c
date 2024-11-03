#include<stdio.h>
#include<string.h>
int KToD(char str[],int k);
int main()
{
	int n,i,k,b[100000],max;
	char a[100000];
	scanf("%d",&n);
	for(i=0;i<n;i++){
		scanf("%s",a);
		getchar();
		scanf("%d",&k);
		b[i]=KToD(a,k);
	}
	max=b[0];
	for(i=0;i<n;i++){
		if(b[i]>max) max=b[i];
	}
	printf("%d",max);
	
	return 0;
}
int KToD(char str[],int k){
	int i,x=1,sum=0;
	for(i=strlen(str)-1;i>=0;i--){
		sum+=(str[i]-'0')*x;
		x=x*k;
	}
	return sum;
}
