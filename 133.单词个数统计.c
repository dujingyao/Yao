#include<stdio.h>
#include<string.h>
int main()
{
	char a[1000];
	int i,n;
	gets(a);
	n=strlen(a);
	int x=0,m=0;
	for(i=0;i<n;i++){
		if(a[i]==' '){
			m=0;
		}else if(m==0){
			m=1;
			x++;
		}
	}
	printf("%d",x);
	
	return 0;
}
