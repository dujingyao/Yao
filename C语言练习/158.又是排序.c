#include<stdio.h>
void psort( int pa, int pb,int pc,int pd);
int main()
{
	int a,b,c,d,*pa=&a,*pb=&b,*pc=&c,*pd=&d;
	scanf("%d %d %d %d",&a,&b,&c,&d);
	psort(*pa,*pb,*pc,*pd);
	printf("%d %d %d %d\n",a,b,c,d);
	
	return 0;
}
void psort( int pa, int pb,int pc,int pd){
	int a[4],i,j,t;
	a[0]=pa,a[1]=pb,a[2]=pc,a[3]=pd;
	for(i=0;i<4;i++){
		for(j=i+1;j<4;j++){
			if(a[i]<a[j]){
				t=a[i];
				a[i]=a[j];
				a[j]=t;
			}
		}
	}
	pa=a[0],pb=a[1],pc=a[2],pd=a[3];
}
