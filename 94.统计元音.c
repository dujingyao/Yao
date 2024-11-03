#include<stdio.h>
int vowel(char ch);
int main()
{
	char ch;
	int count = 0;
	while(scanf("%c", &ch), ch!='\n')
	{
		if(vowel(ch))
			count++;
	}
	printf("%d\n",count);
	
	return 0;
}
int vowel(char ch)
{
	if(ch=='a'||ch=='e'||ch=='i'||ch=='o'||ch=='u'||ch=='A'||ch=='E'||ch=='I'||ch=='O'||ch=='U') return 1;
	else return 0;
}
