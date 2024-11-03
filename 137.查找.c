#include<stdio.h>
#include<string.h>
int main()
{
	char a[200],max;
	int i;
	gets(a);
	max=a[0];
	int	len=strlen(a);
	for(i=0;i<len;i++){
		if(max<=a[i]){
			max=a[i];
		}
	}
	for(i=0;i<len;i++){
		if(a[i]==max) printf("%c(max)",a[i]);
		else printf("%c",a[i]);
	}
	
	
	return 0;
}
