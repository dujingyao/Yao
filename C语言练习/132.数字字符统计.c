#include<stdio.h>
#include<string.h>
int main()
{
	int n,c,x,i;
	char a[1000];
	scanf("%d",&n);
	while(n--){
		c=0;
		scanf("%s",a);
		x=strlen(a);
		for(i=0;i<x;i++){
			if(a[i]>='0'&&a[i]<='9') c++;
		}
		printf("%d\n",c);
	}
	
	return 0;
}
