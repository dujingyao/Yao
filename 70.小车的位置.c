#include<stdio.h>
int main()
{
	int x,y,t,c,p,d;
	x=y=p=d=0;
	while(scanf("%d%d",&t,&c))
	{
		switch(d)
		{
			case 0:y+=(t-p)*10;break;
			case 1:x-=(t-p)*10;break;
			case 2:y-=(t-p)*10;break;
			case 3:x+=(t-p)*10;break;
		}
		if(c==3)
			break;
		if(c==1)
			d++;
		else d--;
		d=(d+4)%4;//保证了c的取值在1~3，+4是避免出现复数
		p=t;
	}
	printf("%d %d\n",x,y);
	return 0;
}
