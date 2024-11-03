#include<stdio.h>
int f(int a,int b,int c);
int main()
{
	int i,j,k,x,a[1000],h=0;
	for(i=1;i<=9;i++){
		for(j=1;j<=9;j++){
			for(k=1;k<=9;k++){
				if(i!=j&&j!=k&&i!=k){
					x=i*100+j*10+k;
					a[h++]=x;
				}
			}
		}
	}
	for(i=0;i<h;i++){
		for(j=i;j<h;j++){
			for(k=j;k<h;k++){
				if(f(a[i],a[j],a[k])) printf("%d %d %d\n",a[i],a[j],a[k]);
			}
		}
	}
	
	return 0;
}
int f(int a,int b,int c){
	int i,d[10]={0},flag=0;
	if(b==a*2&&c==a*3) flag=1;
	while(a){
		i=a%10;
		d[i]++;
		a/=10;
	}
	while(b){
		i=b%10;
		d[i]++;
		b/=10;
	}
	while(c){
		i=c%10;
		d[i]++;
		c/=10;
	}
	for(i=1;i<=9;i++){
		if(d[i]!=1||flag==0) return 0;
	}
	return 1;
}
