#include<stdio.h>
#include<string.h>
int main()
{
	char a[100];
	gets(a);
	int len=strlen(a),i,num=0,m=0;
	for(i=0;i<len;i++){
		if(a[i]>='0'&&a[i]<='9'){
			if(m==0){
				num=a[i]-48;
				m=1;
			}else if(m==1){
				num=num*10+a[i]-48;
			}
		}
	}
	printf("%d",num*2);
	
	return 0;
}
