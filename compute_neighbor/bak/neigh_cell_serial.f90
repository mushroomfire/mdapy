
subroutine build_cell(N, Dim, bin_l, Box_Size, Position, Atom_Cell_List, Cell_Id_List, a, b, c) 
    implicit none

    integer (kind = 8) :: i, icel, jcel, kcel
    integer ( kind = 8 ), intent(in) :: N, Dim, a, b, c
    real (kind = 8) :: xmin, ymin, zmin
    real (kind=8), intent(in) :: bin_l
    integer (kind = 8), dimension(0:Dim-1) :: ncel
    real(kind=8), intent(in), dimension(0:Dim-1, 0:1) :: Box_Size
    real(kind=8), intent(in), dimension(0:N-1, 0:Dim-1) :: Position
    integer (kind = 8), intent(inout), dimension(0:N-1) :: Atom_Cell_List
    !f2py intent(in, out) :: Atom_Cell_List
    integer(kind=8), intent(inout), dimension(0:a-1, 0:b-1, 0:c-1) :: Cell_Id_List 
    !f2py intent(in, out) :: Cell_Id_List 

    xmin = Box_Size(0, 0)
    ymin = Box_Size(1, 0)
    zmin = Box_Size(2, 0)
    ncel = [a, b, c] !floor((Box_Size(:, 1) - Box_Size(:, 0))/bin_l)

    do i = 0, N-1
        icel = floor((Position(i, 0)-xmin)/bin_l)
        jcel = floor((Position(i, 1)-ymin)/bin_l)
        kcel = floor((Position(i, 2)-zmin)/bin_l)
        if (icel < 0) then 
            icel = 0
        else if (icel > ncel(0)-1) then
            icel = ncel(0) - 1
        end if 
        if (jcel < 0) then
            jcel = 0
        else if (jcel > ncel(1)-1) then
            jcel = ncel(1) - 1
        end if 
        if (kcel < 0) then
            kcel = 0
        else if (kcel > ncel(2)-1) then
            kcel = ncel(2) - 1 
        end if 
        Atom_Cell_List(i) = Cell_Id_List(icel, jcel, kcel)
        Cell_Id_List(icel, jcel, kcel) = i 
    end do 

end subroutine


subroutine build_verlet_list_cell(Position, Atom_Cell_List, Cell_Id_List, Mark_List, &
    Verlet_List, bin_l, Rangesqure, N, Dim, &
    Pbc_boundary, Box_Size, a, b, c)
    implicit none
    integer (kind = 8) :: i, j, icel, jcel, kcel, nneiindex, nnei
    integer (kind = 8) :: iicel, jjcel, kkcel, iiicel, jjjcel, kkkcel
    integer ( kind = 8 ), intent(in) :: N, Dim, a, b, c 
    real (kind = 8), intent(in) ::  bin_l, Rangesqure
    real(kind=8), intent(in),dimension(0:N-1, 0:Dim-1) :: Position
    real(kind=8), intent(in),dimension(0:Dim-1, 0:1) :: Box_Size
    real (kind=8), intent(in), dimension(0:Dim-1) :: Pbc_boundary
    integer (kind = 8), dimension(0:Dim-1) :: ncel
    integer(kind = 8), intent(in), dimension(0:N-1) :: Atom_Cell_List
    integer(kind = 8), intent(in), dimension(0:a-1, 0:b-1, 0:c-1) :: Cell_Id_List
    integer(kind = 8), intent(inout), dimension(0:N) :: Mark_List
    !f2py intent(in, out) :: Mark_List
    integer(kind = 8), intent(inout), dimension(0:100*N-1) :: Verlet_List
    !f2py intent(in, out) :: Verlet_List
    real (kind=8), dimension(0:Dim-1) :: Rij
    real (kind = 8) :: Rsqij, xmin, ymin, zmin
    external :: pbc

    xmin = Box_Size(0, 0)
    ymin = Box_Size(1, 0)
    zmin = Box_Size(2, 0)
    ncel = [a, b, c]
    nneiindex = 0
    do i = 0, N-1
        nnei = 0
        icel = floor((Position(i, 0)-xmin)/bin_l)
        jcel = floor((Position(i, 1)-ymin)/bin_l)
        kcel = floor((Position(i, 2)-zmin)/bin_l)
        do iicel = icel-1, icel+1
            do jjcel = jcel-1, jcel+1
                do kkcel = kcel-1, kcel+1
                    iiicel = iicel
                    jjjcel = jjcel
                    kkkcel = kkcel
                    if (iicel < 0) then 
                        iiicel = iicel + ncel(0)
                    else if (iicel > ncel(0)-1) then
                        iiicel = iicel - ncel(0)
                    end if 
                    if (jjcel < 0) then 
                        jjjcel = jjcel + ncel(1)
                    else if (jjcel > ncel(1)-1) then
                        jjjcel = jjcel - ncel(1)
                    end if 
                    if (kkcel < 0) then 
                        kkkcel = kkcel + ncel(2)
                    else if (kkcel > ncel(2)-1) then
                        kkkcel = kkcel - ncel(2)
                    end if 
                    j = Cell_Id_List(iiicel, jjjcel, kkkcel)

                    do while (j > i)
                        Rij = position(i, :) - position(j, :)
                        call pbc(Rij, Pbc_Boundary, Box_Size, Dim)
                        Rsqij = sum(Rij**2)
                        if (Rsqij < Rangesqure) then
                            Verlet_List(nneiindex+nnei) = j
                            nnei = nnei + 1
                        end if 
                        j = Atom_Cell_List(j)
                    end do
                end do 
            end do 
        end do 
    Mark_List(i) = nneiindex
    nneiindex = nneiindex + nnei
    nnei = 0
    end do 
end subroutine

subroutine pbc(Rij, Pbc_Boundary, Box_Size, Dim)
    implicit none

    integer ( kind = 8 ), intent(in) :: Dim
    real(kind=8), intent(in),dimension(0:Dim-1, 0:1) :: Box_Size
    real (kind=8), intent(in), dimension(0:Dim-1) :: Pbc_boundary
    real (kind=8), intent(inout), dimension(0:Dim-1) :: Rij
    !f2py intent(in, out) :: Rij
    real(8) :: L
    integer (kind=8) :: i

    do i = 0, Dim-1
        if (abs(Pbc_Boundary(i) - 1) < 1e-5) then
            L = Box_Size(i, 1) - Box_Size(i, 0)
            Rij(i) = Rij(i) - L*anint(Rij(i)/L)
        end if
    end do
end subroutine