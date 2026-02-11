import java.util.*;
class abc
{
    public static void main()
    {
        Scanner sc = new Scanner(System.in);
        int a[]=sc.nextInt();
        int st=0,en=a0.length,mid,min;
        for (int i=0;i<a.length;i++)
        {
            System.out.println("Enter a number ");
            a[i]=sc.nextInt();
        }
        System.out.println("Enter the number to be found");
        int n=sc.nextInt();
        while(st<=en)
        {
            mid = (st+en)/2;
            if (n==a[mid])
            {
                pos = mid;
                break;
            }
            else if (n<a[mid])
            {
                en= mid-1;
            }
            else 
            {
                st=mid+1;
            }
            if (pos == -1)
            {
                System.out.println("not found");
            }
            else 
            System.out.println("the number is found at "+pos);
            }
        }
    }
}