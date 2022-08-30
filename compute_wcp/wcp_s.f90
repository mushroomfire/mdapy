subroutine get_wcp_real(type_list, verlet_list, data, Zmn, N, max_neigh, N_type)
	
    implicit none
    integer (kind = 8), intent(in) :: N, max_neigh, N_type
    integer(kind = 8), intent(in), dimension(0:N-1, 0:max_neigh-1) :: verlet_list
    integer(kind = 8), intent(in), dimension(0:N-1, 0:1) :: data
    integer(kind = 8), intent(in), dimension(0:N_type-1) :: type_list
    real(kind = 8), dimension(0:N_type-1, 0:N_type-1) :: Zm
    real(kind = 8), intent(inout), dimension(0:N_type-1, 0:N_type-1) :: Zmn
    !f2py intent(in, out) :: Zmn
    integer(kind=8), dimension(:), allocatable :: m_neigh, mtype
    integer(kind=8) :: atomtype1, atomtype11, atomtype2, atomtype22, i, ii, j, jj, jishu
    real(kind=8), dimension(0:N_type-1) :: Alpha_n

Zm = 0
Alpha_n = 0.0
do atomtype1 = 0, N_type-1
    atomtype11 = type_list(atomtype1)
    if (allocated(mtype)) deallocate(mtype)
    allocate(mtype(0:count(data(:, 1)==atomtype11)-1))
    jishu = 0
    do i = 0, N-1
        if (data(i, 1) == atomtype11) then
            mtype(jishu) = i 
            jishu = jishu + 1
        end if
    end do 

    do i = 0, size(mtype)-1
        ii = mtype(i)
        if (allocated(m_neigh)) deallocate(m_neigh)
        allocate(m_neigh(0:count(verlet_list(ii,:)>-1)-1))
        m_neigh = pack(verlet_list(ii, :), verlet_list(ii, :)>-1)
        do atomtype2 = 0, N_type - 1
            atomtype22 = type_list(atomtype2)
            Zm(atomtype11-1, atomtype22-1) = Zm(atomtype11-1, atomtype22-1) + size(m_neigh)
        end do
        do j = 0, size(m_neigh)-1
            jj = m_neigh(j)
            Zmn(atomtype11-1, data(jj, 1)-1) = Zmn(atomtype11-1, data(jj, 1)-1) + 1
        end do
    end do
end do

do i = 0, N_type-1
    ii = type_list(i)
    Alpha_n(i) = count(data(:, 1)==ii)*1.0/N
end do

do i = 0, N_type-1
    Zm(i, :) = Zm(i, :) * Alpha_n
end do 

Zmn = 1 - Zmn/Zm
end subroutine

