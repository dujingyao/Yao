#include<stdio.h>
#include<string.h>
int main()
{
	char a[1000];
	gets(a);
	int i,x=0,j,len=strlen(a);
	for(i=0;i<len;i++){
		if((a[i]>='1'&&a[i]<='9')||(i==0&&a[i]=='0')||(i>0&&a[i]=='0'&&(a[i-1]<'0'||a[i-1]>'9'))){
			x++;
            for(j=i+1;j<len;j++){
                if(a[j]<'0'||a[j]>'9'){
                    i=j;
                    break;
                }
                if(j==len-1){
                    i=j;
                }
            }
			}
		}
	printf("%d\n",x);
	return 0;
}
