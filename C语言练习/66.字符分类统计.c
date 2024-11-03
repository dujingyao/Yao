#include<stdio.h>
int main()
{
	char ch;
	int a=0,b=0,c=0;
	while(ch=getchar(),ch!='\n')
	{
		if((ch>='a'&&ch<='z')||(ch>='A'&&ch<='Z')) a++;
		else if(ch>='0'&&ch<='9') b++;
		else c++;
	}
	printf("%d\n%d\n%d\n",a,b,c);
	
	return 0;
}
