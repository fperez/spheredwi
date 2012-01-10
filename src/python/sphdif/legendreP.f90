subroutine legP_f(nMax,mu,P)
!
! Subroutine to generate Legendre polynomials P_n(mu)
! for n = 0,1,...,nMax with given mu.
!
  implicit none
  integer, intent(in) :: nMax
  integer :: n
  real, intent(in) :: mu
  real, intent(out), dimension(0:nMax) :: P
!
  P(0) = 1.0
  P(1) = mu
  do n = 1, nMax-1
    P(n+1) = ((2.0*n + 1.0) * mu * P(n) - n*P(n-1))/(n + 1.0)
  end do

end subroutine legP_f
