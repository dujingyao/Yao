#include<stdio.h>
int HmsToS(int h,int m,int s)
{
	int x=h*3600+m*60+s;
	return x;
}
void PrintTime(int s)
{
	int h,m,n;
	h=(int)s/3600;
	m=(int)s%3600/60;
	n=(int)s%3600%60;
	printf("%02d:%02d:%02d\n",h,m,n);
}
int main()
{
	
	return 0;
}
