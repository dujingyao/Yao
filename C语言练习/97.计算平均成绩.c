#include<stdio.h>
int getScore(char g)
{
	int x;
	if(g=='A') x=95;
	else if(g=='B') x=85;
	else if(g=='C') x=75;
	else if(g=='D') x=65;
	else if(g=='E') x=40;
	else x=0;
	return x;
}
int main()
{
	char ch;
	int i=0;
	double g,y,h=0;
	while((ch=getchar())!='\n')
	{
		i++;
		y=getScore(ch);
		h=y+h;
	}
	g=1.0*h/i;
	printf("%.1f",g);
	
	return 0;
}
