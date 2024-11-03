#include<stdio.h>
int main()
{
	int n,m;
	scanf("%d %d",&n,&m);
	int a[n],b[m],x=0;
	int i,j=0,t;
	for(i=0;i<n;i++){
		scanf("%d",&a[i]);
	}
	for(i=0;i<m;i++){
		scanf("%d",&b[i]);
	}
	for(i=0;i<n;i++){
		if(a[i]==b[j]){
			j++;
			x++;
		}else{
			j=0;
			x=0;
		}
		t=i;
		if(j>=m) break;
	}
	if(x==m) printf("%d",t-m+1);
	else printf("No Answer");
	
	return 0;
}
