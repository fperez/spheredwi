subroutine even_kernel_f(mu, value, nMax)
!
! Subroutine to calculate the even kernel at mu
!
  implicit none
  integer, intent(in)     :: nMax
  real, intent(in)        :: mu
  integer                 :: n
  real, dimension(0:nMax) :: P, coefs
  real                    :: pi  
  real, intent(out) :: value
!
  pi = 3.141592653589
!
! Calculate legendre polynomials at mu
  P = 0.0
  call legP_f(nMax,mu,P)

! Calculate coefficients
  coefs = 0.0
  do n=0, nMax, 2
    coefs(n) = (2.0 * n + 1.0)
  end do
  coefs = coefs / (4.0 * pi)

  value = dot_product(coefs, P)
!  
end subroutine even_kernel_f
