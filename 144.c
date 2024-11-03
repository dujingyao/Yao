#include<stdio.h>
#include<string.h>
void dToK(int n, int k, char str[]);
int main()
{
	int n;
	char str[10000];
	scanf("%d",&n);
	dToK(n,2,str);
	dToK(n,3,str);
	dToK(n,7,str);
	dToK(n,8,str);
	
	return 0;
}
void dToK(int n, int k, char str[]){
	int i;
	for(i=0;n>0;i++){
		str[i]=n%k+'0';
		n=n/k;
	}
	str[i]='\0';
	int len=strlen(str);
	for(i=len-1;i>=0;i--){
		printf("%c",str[i]);
	}
	printf('\n');
}
