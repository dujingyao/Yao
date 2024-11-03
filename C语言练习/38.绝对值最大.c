#include<stdio.h>
#include<math.h>
int main()
{
	int a,b,c;
	scanf("%d %d %d",&a,&b,&c);
	int x=abs(a),y=abs(b),z=abs(c);
	if(x>=y&&x>=z)
		printf("%d\n",a);
	else if(y>=x&&y>=z)
		printf("%d\n",b);
	else if(z>=x&&z>=y)
		printf("%d\n",c);
	
	return 0;
}
