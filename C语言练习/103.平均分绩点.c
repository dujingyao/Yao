#include<stdio.h>
double y(int a)
{
	int j;
	if(a>60||a==60) j=(a-50)/10;
	else j=0;
	return j;
}
int main()
{
	int a,b,n;
	int x=0,h=0;//h是绩点，x是学分
	double f;
	scanf("%d",&n);
	while(n>0)
	{
		n--;
		scanf("%d %d",&a,&b);
		x=x+a;
		h=h+y(b)*a;
	}
	f=1.0*h/x;
	printf("%.1f",f);
	return 0;
}
