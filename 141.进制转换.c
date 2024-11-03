#include<stdio.h>
void convert(int n, char str[]);
int main()
{
	int n;
	char str[40];
	scanf("%d",&n);
	convert(n,str);
	
	return 0;
}
void convert(int n,char str[]){
	int len=0;
	if(n==0) printf("0");
	else{
		while(n>0){
			str[len]=n%2+'0';
			len++;
			n/=2;
		}
		str[len]='\0';
		int i;
		for(i=len-1;i>=0;i--) printf("%c",str[i]);
	}
}
