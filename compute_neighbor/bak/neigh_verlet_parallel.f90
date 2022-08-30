subroutine neigh(Position, N, Pbc_boundary, Box_Size, R_Cutoff, Skin, Mark1, Mark2, Neigh_List, Dim, n_thread)
    use omp_lib
    implicit none

    integer ( kind = 4 ), intent(in) :: N, Dim, n_thread
    integer (kind = 8) :: L, i, j, L1, loc
    real(kind=8), intent(in),dimension(0:N-1, 0:Dim-1) :: Position 
    real(kind=8), intent(in),dimension(0:Dim-1, 0:1) :: Box_Size
    real (kind=4), intent(in), dimension(0:Dim-1) :: Pbc_boundary
    real ( kind = 8 ), intent(in) :: R_Cutoff, Skin
    real (kind = 8) :: Range, Rangesqure, Rsqij
    real (kind=8), dimension(0:Dim-1) :: Rij
    integer ( kind = 8 ), intent(inout), dimension(0:N-1) :: Mark1, Mark2
    !f2py intent(in, out) :: Mark1, Mark2
    integer ( kind = 8 ), intent(inout), dimension(0:N-1, 0:100-1) :: Neigh_List
    !f2py intent(in, out) :: Neigh_List
    !integer (kind = 8), dimension(0:N-1) :: Advance
    external :: pbc

    call OMP_SET_NUM_THREADS(n_thread)
    !write ( *, '(a,i8)' ) '  The number of threads available is:    ', omp_get_max_threads ( )
    Range = R_Cutoff + Skin
    Rangesqure = Range * Range
    !Mark1 = 0
    !Mark2 = 0
    Neigh_List = -1
    !$omp parallel &
    !$omp shared ( Position, N, Pbc_Boundary, Box_Size, Rangesqure, Neigh_List) &
    !$omp private ( i, j, L, Rij, Rsqij )
    !$omp do
    do i = 0, N-1
        L = 0
        do j = i+1, N-1
            Rij = Position(i, :) - Position(j, :)
            call pbc(Rij, Pbc_Boundary, Box_Size, Dim)
            Rsqij = Rij(0)**2 + Rij(1)**2 + Rij(2)**2
            if (Rsqij < Rangesqure) then
                Neigh_List(i, L) = j
                L = L + 1
            end if
        end do
        L = 0
    end do
    !$omp end do
    !$omp end parallel
end subroutine

subroutine pbc(Rij, Pbc_Boundary, Box_Size, Dim)
    implicit none

    integer ( kind = 4 ), intent(in) :: Dim
    real(kind=8), intent(in),dimension(0:Dim-1, 0:1) :: Box_Size
    real (kind=4), intent(in), dimension(0:Dim-1) :: Pbc_boundary
    real (kind=8), intent(inout), dimension(0:Dim-1) :: Rij
    !f2py intent(in, out) :: Rij
    real(8) :: L
    integer (kind=4) :: i

    do i = 0, Dim-1
        if (abs(Pbc_Boundary(i) - 1) < 1e-5) then
            L = Box_Size(i, 1) - Box_Size(i, 0)
            Rij(i) = Rij(i) - L*anint(Rij(i)/L)
        end if
    end do
end subroutine