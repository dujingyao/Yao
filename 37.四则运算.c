#include<stdio.h>
#include<math.h>
int main()
{
	double a,b,c;
	char x;
	int y=0;
	scanf("%lf %c %lf",&a,&x,&b);
	switch (x) {
	case '+':
		c=a+b;
		break;
	case '-':
		c=a-b;
		break;
	case '*':
		c=a*b;
		break;
	case '/':
		if(fabs(b)<=1e-10)
			y=1;
		else c=a/b;
		break;
	default:
		y=1;
		break;
	}
	if(y==0)
		printf("%.2lf",c);
	else
		printf("Wrong input!\n");
	
	return 0;
}
