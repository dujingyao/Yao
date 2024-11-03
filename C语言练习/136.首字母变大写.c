#include<stdio.h>
#include<string.h>
int main()
{
	char a[100];
	int i;
	gets(a);
	int len=strlen(a);
	for(i=0;i<len-1;i++){
		if(a[0]<='z'&&a[0]>='a') a[0]=a[0]-32;
		if(a[i]==' '&&a[i+1]>='a'&&a[i+1]<='z') a[i+1]=a[i+1]-32;
	}
	for(i=0;i<len;i++){
		printf("%c",a[i]);
	}
	return 0;
}
