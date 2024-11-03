#include<stdio.h>
#include<string.h>
int main()
{
	int n,i;
	scanf("%d",&n);
	getchar();
	char a[1000],min[1000];
	gets(a);
	strcpy(min,a);
	for(i=1;i<n;i++){
		gets(a);
		if(strlen(a)<strlen(min)) strcpy(min,a);
	}
	printf("%s\n",min);
	
	return 0;
}
