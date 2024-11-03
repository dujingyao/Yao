#include<stdio.h>
int main()
{
	char ch;
	while(scanf("%c",&ch),ch!='@')
	{
		if(ch>='A'&&ch<='Z') ch+=32;
		
		if(ch>='a'&&ch<='z') ch+=1;
		else if(ch=='z') ch='a';
		printf("%c",ch);
	}
	
	return 0;
}
