#include<stdio.h>
double funF(double x)
{
	double F;
	if(x>3||x==3) F=x+x-3+1;
	else if(x>-1&&x<3) F=4;
	else if(x<-1||x==-1) F=2-x-x;
	return F;
}
double funG(double x)
{
	double G;
	G=x*x-3*x;
	return G;
}
int main()
{	
	return 0;
}
