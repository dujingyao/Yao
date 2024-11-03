#include<stdio.h>
double CancelFee(double price)
{
	double y;
	y=price*0.05;
	double h=y;
	while(y>=1)
	{
		y--;
	}
	if(y<0.25) h=h-y;
	if((y>0.25||y==0.25)&&y<0.75) h=h-y+0.5;
	if(y>0.75||y==0.75) h=h-y+1;
	return h;
}
int main()
{
	
	return 0;
}
