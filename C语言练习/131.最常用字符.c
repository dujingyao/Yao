#include<stdio.h>
#include<string.h>
int main()
{
	int i,b[27]={0};
	char a[100];
	gets(a);
	int len=strlen(a);
	for(i=0;i<len;i++){
		if(a[i]>='A'&&a[i]<='Z'){
			a[i]=a[i]+32;
		}
		b[a[i]-'a'+1]++;
	}
	int max=b[0],temp;
	for(i=1;i<=26;i++){
		if(max<b[i]){
			max=b[i];
			temp=i;
		}
	}
	printf("%c",temp+'a'-1);
	
	return 0;
}
