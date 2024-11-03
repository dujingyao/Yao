#include<stdio.h>
#include<string.h>
int main()
{
	char a[50];
	gets(a);
	int x=0,i,len=strlen(a);
	if(a[0]>='a'&&a[0]<='z'||a[0]>='A'&&a[0]<='Z'||a[0]=='_') x=1;
	if(x==1){
		for(i=0;i<len;i++){
			if(a[i]>='a'&&a[i]<='z'||a[i]>='A'&&a[i]<='Z'||a[i]>='0'&&a[i]<='9'||a[i]=='_') x++;
		}
	}
	if(x==len+1) printf("yes");
	else printf("no");
	
	return 0;
}
