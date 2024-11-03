#include<stdio.h>
#include<string.h>
int main()
{
	int t,i,n,k;
	scanf("%d",&t);
	char a[101];
	while(t--){
		scanf("%s",a);
		scanf("%d",&n);
		for(i=0;i<strlen(a);i++){
			if(a[i]=='.'){
				k=i;
				break;
			}
		}
		if((k+n)<strlen(a)) printf("%c\n",a[k+n]);
		else printf("0\n");
	}
	
	
	return 0;
}
