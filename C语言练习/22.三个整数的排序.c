#include<stdio.h>
int main()
{
	int x,y,z;
	scanf("%d %d %d",&x,&y,&z);
	if(x>y){
		if(x>z){
			if(y>z){
				printf("%d %d %d",x,y,z);
			}else printf("%d %d %d",x,z,y);
		}else printf("%d %d %d",z,x,y);
	}else if(x<z){
				if(y<z){printf("%d %d %d",z,y,x);
				}else printf("%d %d %d",y,z,x);
	}else printf("%d %d %d",y,x,z);
	
	return 0;
}
